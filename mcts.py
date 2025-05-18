import numpy as np
import math
import chess
import torch

# Configuration imports
from config import CPUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, DEVICE, ACTION_SIZE, NUM_SIMULATIONS

# Utility imports - ensure utils.py is correct and these functions exist
from utils import get_legal_actions_mask, action_idx_to_move, board_to_tensor, move_to_action_idx

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    """
    def __init__(self, parent, prior_p: float, board_state_key: str):
        self.parent = parent  # Parent MCTSNode
        self.children = {}    # Dictionary mapping action_idx to child MCTSNode
        self.N = 0            # Visit count
        self.W = 0.0          # Total action value (sum of values from simulations passing through this node)
        self.Q = 0.0          # Mean action value (W / N)
        self.P = prior_p      # Prior probability of selecting this node's incoming action (from NN policy)
        self.board_state_key = board_state_key # Typically a FEN string or Zobrist hash to identify board state

class MCTS:
    """
    Monte Carlo Tree Search implementation.
    """
    def __init__(self, model: torch.nn.Module, initial_board: chess.Board):
        self.model = model
        self.model.eval()  # Ensure neural network is in evaluation mode
        
        # Transposition table for MCTS nodes to reuse computation for identical board states
        self.tree_nodes = {} 
        
        self.root_board_fen = initial_board.fen() # Store FEN of the root board MCTS was initialized with
        self.root = self._get_or_create_node(initial_board, parent=None, prior_p=1.0) # prior_p for root is nominal

    def _get_board_key(self, board: chess.Board) -> str:
        # Using FEN as a key. Could use board._transposition_key() for Zobrist hash.
        return board.fen()

    def _get_or_create_node(self, board: chess.Board, parent: MCTSNode | None, prior_p: float) -> MCTSNode:
        board_key = self._get_board_key(board)
        if board_key not in self.tree_nodes:
            self.tree_nodes[board_key] = MCTSNode(parent, prior_p, board_key)
        # If node exists, ensure its parent is updated if we reach it via a different path (though less common with strict tree)
        # For this MCTS structure, we typically create a new node or reuse an existing one fully.
        # If reusing, its P might be from a previous expansion. For root, P is nominal.
        # If it's not the root being created here, parent and prior_p are from a child selection.
        node = self.tree_nodes[board_key]
        if parent is not None and node.parent is None and node is not self.root : # if it was an old root
            node.parent = parent # Re-attach if it became part of a deeper search
        return node

    def _select_child(self, node: MCTSNode, board: chess.Board) -> tuple[int | None, MCTSNode | None]:
        """
        Selects a child node using the PUCT formula.
        `board` is the board state corresponding to `node`.
        Returns (action_idx, child_node) or (None, None) if no valid child.
        """
        best_score = -float('inf')
        best_action_idx = None
        best_child_node = None

        if not node.children: # Should not happen if called on non-leaf, but as safeguard
            return None, None

        # Get a mask of legal actions from the current board state
        # This is critical: PUCT should only be applied to children corresponding to legal moves.
        legal_mask_np = get_legal_actions_mask(board).cpu().numpy()

        for action_idx, child_node in node.children.items():
            if not (0 <= action_idx < ACTION_SIZE and legal_mask_np[action_idx]):
                # This child corresponds to an action that is currently illegal. Skip it.
                # This can happen if the tree has nodes for actions that were legal from a slightly
                # different (but same FEN key) path, or if legal_mask is very strict.
                continue

            # PUCT formula: Q(s,a) + U(s,a)
            # U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
            # sum_b N(s,b) is node.N (parent's visit count)
            u_score = child_node.Q + CPUCT * child_node.P * \
                      (math.sqrt(node.N) / (1 + child_node.N))
            
            if u_score > best_score:
                best_score = u_score
                best_action_idx = action_idx
                best_child_node = child_node
        
        return best_action_idx, best_child_node

    def _expand_and_evaluate(self, node: MCTSNode, board: chess.Board):
        """
        Expands a leaf node: runs NN, creates children, gets value.
        `board` is the board state corresponding to `node`.
        Returns the value of the current state from NN or game outcome.
        """
        game_over, outcome_val_for_current_player = self._check_game_over_and_get_value(board)

        if game_over:
            return outcome_val_for_current_player # Value is from game result

        # If not game over, use Neural Network to get policy and value
        # board_to_tensor creates canonical board representation
        board_tensor_gpu = board_to_tensor(board).to(DEVICE)
        with torch.no_grad():
            policy_logits, value_from_nn = self.model(board_tensor_gpu)
        
        # Value is from the NN's perspective (current player at 'board')
        value = value_from_nn.item() 
        
        policy_probs_gpu = torch.softmax(policy_logits, dim=1).squeeze(0)
        
        # Add Dirichlet noise to root node's children's priors for exploration in self-play
        # This should happen *before* using these policy_probs to create children if node is root
        # For simplicity in this flow, we can apply noise after initial P is set if it's root.
        # Alternative: apply noise to policy_probs_gpu before iterating.
        
        # Mask illegal moves from policy probabilities before creating children
        legal_mask = get_legal_actions_mask(board).to(DEVICE) # Ensure mask is on same device as policy
        masked_policy_probs = policy_probs_gpu * legal_mask
        sum_masked_policy_probs = torch.sum(masked_policy_probs)

        if sum_masked_policy_probs > 1e-6: # Normalize after masking
            masked_policy_probs /= sum_masked_policy_probs
        else:
            # All legal moves had zero prior prob, or no legal moves (should be caught by game_over)
            # This is a problematic state for NN policy.
            # Fallback: if any legal moves, assign uniform prob among them.
            # print(f"Warning (_expand_and_evaluate): NN policy sum is zero for legal moves on board:\n{board}")
            num_legal = torch.sum(legal_mask).item()
            if num_legal > 0:
                masked_policy_probs = legal_mask.float() / num_legal
            # If no legal moves, game_over should have caught it.

        policy_probs_cpu = masked_policy_probs.cpu().numpy()

        # Create child nodes
        for action_idx in range(ACTION_SIZE):
            if policy_probs_cpu[action_idx] > 0: # Only create children for actions with some probability
                # We don't need to simulate the move here to get child_board_key yet.
                # The child MCTSNode is created with prior_p from NN.
                # Its board_state_key will be set when it's actually visited and becomes 'current_board'.
                # However, for transposition table, we'd ideally use the key of the *resulting* state.
                # For now, node children are just indexed by action_idx.
                # When a child is selected, we make the move, then get/create the node for the new state.
                node.children[action_idx] = MCTSNode(parent=node,
                                                     prior_p=policy_probs_cpu[action_idx],
                                                     board_state_key="PENDING") # Key set upon visit
        
        # Apply Dirichlet noise if this expanded node is the root of the MCTS search
        if node == self.root and node.N == 0 : # Apply only on first expansion of root
             self._add_dirichlet_noise_to_children(node, board, policy_probs_cpu)


        return value # Return value from NN for non-terminal state

    def _add_dirichlet_noise_to_children(self, node: MCTSNode, board: chess.Board, original_policy_probs: np.ndarray):
        """Adds Dirichlet noise to the prior probabilities of the (legal) children of a node."""
        
        # Identify indices of children that correspond to legal moves
        legal_action_indices = []
        for action_idx, child_node in node.children.items():
            # Check if action_idx truly corresponds to a legal move on 'board'
            # This relies on action_idx_to_move being robust.
            # Or, more simply, iterate board.legal_moves and find their mapped action_idx.
            move_obj = action_idx_to_move(action_idx, board)
            if move_obj is not None: # If it's a valid move mapping and legal
                legal_action_indices.append(action_idx)
        
        if not legal_action_indices:
            return

        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_action_indices))
        
        for i, action_idx in enumerate(legal_action_indices):
            child_node = node.children.get(action_idx)
            if child_node: # Should exist as we just created them
                 # The original child_node.P was set from (masked and normalized) policy_probs_cpu
                 # We now adjust this P with Dirichlet noise.
                 child_node.P = (1 - DIRICHLET_EPSILON) * child_node.P + DIRICHLET_EPSILON * noise[i]


    def _backup(self, node: MCTSNode, value: float):
        """
        Backpropagates the simulation result (value) up the tree.
        Value is from the perspective of the player whose turn it was at the *end* of the simulation path.
        """
        current_node = node
        current_value = value # Value is from perspective of player AT THE EXPANDED NODE
        
        while current_node is not None:
            current_node.N += 1
            current_node.W += current_value 
            current_node.Q = current_node.W / current_node.N
            
            # Value needs to be flipped for the parent, as it's from opponent's perspective.
            current_value *= -1 
            current_node = current_node.parent

    def _check_game_over_and_get_value(self, board: chess.Board) -> tuple[bool, float]:
        """
        Checks if the game is over at the current board state.
        Returns (is_game_over, value_for_current_player).
        Value: +1 if current player wins, -1 if loses, 0 for draw.
        """
        if board.is_checkmate():
            # Current player (board.turn) is checkmated, so they lose.
            return True, -1.0 
        if board.is_stalemate() or \
           board.is_insufficient_material() or \
           board.is_seventyfive_moves() or \
           board.is_fivefold_repetition() or \
           board.can_claim_draw(): # General draw claim
            return True, 0.0
        return False, 0.0 # Game not over, nominal value (will be replaced by NN)

    def search(self, current_board: chess.Board, num_simulations: int) -> tuple[dict, chess.Move | None]:
        """
        Performs MCTS simulations from the current_board state.
        Updates the MCTS tree.
        Returns a policy dictionary {move_uci: probability} and the chosen chess.Move.
        """
        
        # Ensure the MCTS root corresponds to the current_board state
        current_board_key = self._get_board_key(current_board)
        if self.root.board_state_key != current_board_key:
            # We've moved; find or create the new root from the tree.
            # This happens if MCTS object is reused across moves.
            # print(f"MCTS re-rooting from {self.root.board_state_key} to {current_board_key}")
            self.root = self._get_or_create_node(current_board, parent=None, prior_p=1.0)
            self.root_board_fen = current_board_key # Update what MCTS considers root FEN

        for _ in range(num_simulations):
            node = self.root
            sim_board = current_board.copy() # Board for this simulation path
            
            # --- 1. Select ---
            # Traverse the tree until a leaf node (not yet expanded or terminal) is reached.
            while node.children: # While node has been expanded
                action_idx, next_node_candidate = self._select_child(node, sim_board)
                
                if action_idx is None or next_node_candidate is None:
                    # No valid child to select (e.g., all children illegal, or PUCT yielded nothing)
                    # This node effectively becomes a leaf for this simulation path.
                    break 
                
                selected_move = action_idx_to_move(action_idx, sim_board)
                if selected_move is None: # Selected action is illegal on sim_board
                    # This means _select_child or action_idx_to_move has an issue,
                    # or legal_mask in _select_child was incorrect.
                    # print(f"CRITICAL MCTS Error: Selected action_idx {action_idx} is illegal during SELECTION for board:\n{sim_board}")
                    break # Treat as leaf
                
                sim_board.push(selected_move)
                # The next_node_candidate is just a placeholder from parent's children dict.
                # The actual node for sim_board state is fetched/created here.
                node = self._get_or_create_node(sim_board, parent=node, prior_p=next_node_candidate.P)
                # Update child's true parent if we are re-using a node from transposition table
                if node.parent != next_node_candidate.parent: # Should be node.parent != node ?
                    node.parent = next_node_candidate.parent # Ensure parent link is correct
                                 
            # --- 2. Expand & Evaluate ---
            # If node.N > 0, it means it was visited before (potentially in earlier sim or from TT).
            # If it's a true leaf (never expanded), node.children will be empty.
            value_for_backup = 0.0
            if not self._check_game_over_and_get_value(sim_board)[0]: # If game not over at this leaf
                if not node.children: # If truly unexpanded leaf
                    # Expand the node, get value from NN
                    value_for_backup = self._expand_and_evaluate(node, sim_board)
                else: # Already expanded (e.g. from transposition table, or a previous simulation in THIS search call)
                      # but selection phase might have stopped here if all children were illegal.
                      # Re-evaluate with NN if it's not terminal.
                    _g, nn_val = self._check_game_over_and_get_value(sim_board)
                    if not _g:
                        board_tensor_gpu = board_to_tensor(sim_board).to(DEVICE)
                        with torch.no_grad():
                            _, value_from_nn = self.model(board_tensor_gpu)
                        value_for_backup = value_from_nn.item()
                    else:
                         value_for_backup = nn_val # Game ended
            else: # Game is over at this leaf
                _, value_for_backup = self._check_game_over_and_get_value(sim_board)

            # --- 3. Backup ---
            self._backup(node, value_for_backup)

        # After all simulations, get action probabilities from root's visit counts
        return self._get_final_action_probs(self.root, current_board)


    def _get_final_action_probs(self, root_node: MCTSNode, board: chess.Board, temp=0.01) -> tuple[dict, chess.Move | None]:
        """
        Calculates the final policy and selects a move after simulations.
        `temp` close to 0 means more deterministic (greedy) move selection.
        Returns: policy_dict {move_uci: probability_float}, chosen_chess.Move_object
        """
        
        game_is_over, _ = self._check_game_over_and_get_value(board)
        if game_is_over:
            return {}, None

        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action_idx, child_node in root_node.children.items():
            move_obj_check = action_idx_to_move(action_idx, board) # Check if this action is legal now
            if move_obj_check is not None:
                counts[action_idx] = child_node.N
        
        policy_for_training = {} # Initialize policy dictionary for training

        if np.sum(counts) == 0:
            # print(f"Warning (_get_final_action_probs): MCTS visit counts are all zero for board:\n{board}")
            legal_moves = list(board.legal_moves)
            if not legal_moves: return {}, None
            
            chosen_move_obj = np.random.choice(legal_moves)
            # Create policy_for_training: uniform over legal moves
            num_legal = len(legal_moves)
            for m_obj in legal_moves:
                # We use the UCI string as the key for the policy dictionary sent for training data
                policy_for_training[m_obj.uci()] = 1.0 / num_legal
            return policy_for_training, chosen_move_obj

        probs_vector = np.zeros_like(counts) # Ensure it's initialized
        if temp == 0:
            best_action_idx = np.argmax(counts)
            if np.sum(counts) > 0: probs_vector[best_action_idx] = 1.0
        else:
            adjusted_counts = counts**(1.0 / temp)
            sum_adjusted_counts = np.sum(adjusted_counts)
            if sum_adjusted_counts > 1e-9 : # Avoid division by zero
                probs_vector = adjusted_counts / sum_adjusted_counts
            else: # Fallback if all adjusted counts are zero (e.g. counts were tiny and temp very high)
                if np.sum(counts) > 0: # If original counts existed
                    best_action_idx = np.argmax(counts)
                    probs_vector[best_action_idx] = 1.0


        # Populate policy_for_training using probs_vector
        for action_idx, prob_val in enumerate(probs_vector):
            if prob_val > 0:
                # Convert action_idx back to a move object, then to UCI
                # We need the board context here to ensure the move makes sense,
                # though action_idx_to_move already checks legality.
                # For policy_for_training, we primarily need the UCI of the action index.
                # A simpler way if _INV_ACTION_MAP was still global in utils:
                #   move_obj_from_map = _INV_ACTION_MAP[action_idx]
                # But _INV_ACTION_MAP is gone. action_idx_to_move decodes it.
                # However, action_idx_to_move requires a board. For the policy dict,
                # we are mapping what the NN *could* output.
                #
                # The most robust way is if action_idx_to_move can give us a "raw" move
                # without full board validation, just for the purpose of getting its UCI.
                # Or, iterate through `root_node.children` which are already validated to some extent.

                # Let's reconstruct the move if possible to get its UCI for the policy dict.
                # This part is tricky without the global _INV_ACTION_MAP.
                # The `action_idx_to_move` *requires* a board.
                # For the policy target, we are creating a distribution over the *entire action space*.
                # The key of the policy dictionary MUST be the UCI string that `get_policy_target`
                # in `utils.py` will then map using `move_to_action_idx`.
                #
                # Best approach for policy_for_training keys:
                # When `get_policy_target` receives this dict, it parses each UCI key
                # using `board.parse_uci()` and then `move_to_action_idx()`.
                # So, the UCI string needs to be standard.
                #
                # Let's iterate through children of root_node to get their UCIs if they were visited
                # This is safer for constructing policy_for_training.
                pass # Will reconstruct below more carefully.

        # Reconstruct policy_for_training based on children that were actually explored and legal
        # This ensures policy_for_training keys are valid UCIs of moves considered.
        temp_policy_probs_for_training = {}
        total_prob_sum_for_training = 0
        for action_idx, child_node_val in root_node.children.items():
            if counts[action_idx] > 0 : # If this child was visited (and thus considered legal at some point)
                move_obj = action_idx_to_move(action_idx, board) # Re-check legality on current board
                if move_obj:
                    # The probability for this move in the training target is from probs_vector
                    temp_policy_probs_for_training[move_obj.uci()] = probs_vector[action_idx]
                    total_prob_sum_for_training += probs_vector[action_idx]
        
        # Normalize policy_for_training if needed
        if total_prob_sum_for_training > 1e-6:
            for uci_key in temp_policy_probs_for_training:
                policy_for_training[uci_key] = temp_policy_probs_for_training[uci_key] / total_prob_sum_for_training
        elif temp_policy_probs_for_training: # if dict not empty but sum is too small
             # print("Warning: Sum of training policy probs very small after filtering. Using uniform over considered.")
             num_considered = len(temp_policy_probs_for_training)
             for uci_key in temp_policy_probs_for_training:
                 policy_for_training[uci_key] = 1.0 / num_considered

        # --- Select the actual move to play in the game ---
        candidate_indices = np.where(probs_vector > 0)[0]
        valid_action_indices_for_choice = []
        final_probs_for_choice = []

        for idx_choice in candidate_indices:
            move_choice = action_idx_to_move(idx_choice, board)
            if move_choice is not None: # Is legal on current board
                valid_action_indices_for_choice.append(idx_choice)
                final_probs_for_choice.append(probs_vector[idx_choice])
        
        if not valid_action_indices_for_choice:
            # print(f"CRITICAL Warning (_get_final_action_probs): No valid action to choose from policy for board:\n{board}")
            legal_moves = list(board.legal_moves)
            if not legal_moves: return policy_for_training, None
            chosen_move_obj = np.random.choice(legal_moves)
            if not policy_for_training : # If policy was empty, make it reflect this choice
                policy_for_training = {m.uci(): (1.0/len(legal_moves) if m == chosen_move_obj else 0.0) for m in legal_moves}
            return policy_for_training, chosen_move_obj

        final_probs_for_choice_sum = sum(final_probs_for_choice)
        if final_probs_for_choice_sum > 1e-6:
            normalized_final_probs = [p / final_probs_for_choice_sum for p in final_probs_for_choice]
        else:
            normalized_final_probs = [1.0 / len(valid_action_indices_for_choice)] * len(valid_action_indices_for_choice)

        chosen_action_idx = np.random.choice(valid_action_indices_for_choice, p=normalized_final_probs)
        chosen_move_obj = action_idx_to_move(chosen_action_idx, board)

        # Ensure policy_for_training is not empty if a move was chosen
        if chosen_move_obj and not policy_for_training and board.legal_moves:
            # This can happen if the probs_vector logic led to an empty policy_for_training
            # but a move was still chosen (e.g. from fallback). Reconstruct based on legal moves.
            # print("Warning: policy_for_training was empty but a move was chosen. Reconstructing policy.")
            num_legal = len(list(board.legal_moves))
            policy_for_training = {m.uci(): 1.0 / num_legal for m in board.legal_moves}


        return policy_for_training, chosen_move_obj



if __name__ == '__main__':
    # This import is here because neural_net might import utils, which might have issues if mcts is imported first by utils.
    # However, utils should not import mcts.
    from neural_net import ChessNN 
    
    _initialize_action_maps() # Crucial for utils functions called by MCTS and this test
    
    print(f"Testing MCTS on device: {DEVICE}")
    # Use a dummy model for testing MCTS logic
    # The model's quality affects MCTS decisions but not the MCTS mechanics themselves.
    dummy_nn_model = ChessNN().to(DEVICE) # Create an instance of your NN
    dummy_nn_model.eval()

    initial_board = chess.Board()
    print("Initial board for MCTS test:\n", initial_board)

    # Create MCTS instance with the model and initial board state
    mcts_agent = MCTS(model=dummy_nn_model, initial_board=initial_board.copy())
    
    print(f"Running MCTS search with {NUM_SIMULATIONS} simulations...")
    # The search method takes the current board state from which to search
    # For the first move, this is the same as initial_board.
    # Result is: dict of {move_uci: probability}, chosen_move_object
    mcts_policy_dict, chosen_game_move = mcts_agent.search(initial_board.copy(), NUM_SIMULATIONS) 
    
    print("\n--- MCTS Search Results ---")
    if chosen_game_move:
        print(f"Chosen move by MCTS: {chosen_game_move.uci()}")
        print("Policy distribution from MCTS (top 5 or all if fewer):")
        if mcts_policy_dict:
            sorted_policy = sorted(mcts_policy_dict.items(), key=lambda item: item[1], reverse=True)
            for i, (move_uci, prob) in enumerate(sorted_policy):
                if i < 5:
                    print(f"  {move_uci}: {prob:.4f}")
            if not sorted_policy : print(" (Policy dictionary is empty)")
        else:
            print(" (Policy dictionary from MCTS is empty or None)")
        
        # Example of how to use the policy_dict for training target (from utils)
        # This would typically happen in the self-play loop.
        # policy_target_tensor = get_policy_target(initial_board, mcts_policy_dict, initial_board.turn)
        # print("\nSample policy target tensor for training (first 10 elements):")
        # print(policy_target_tensor[:10])
        # print(f"Policy target sum: {torch.sum(policy_target_tensor).item()}")

    else:
        print("MCTS did not choose a move.")
        if initial_board.is_game_over():
            print("Game is already over:", initial_board.outcome())

    print("\nMCTS test finished.")