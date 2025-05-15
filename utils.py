import chess
import numpy as np
import torch
from config import INPUT_SHAPE, ACTION_SIZE, DEVICE # ACTION_SIZE must be 4672 for this

# Ensure ACTION_SIZE is 4672 for this Lc0-style mapping
if ACTION_SIZE != 4672:
    raise ValueError("ACTION_SIZE in config.py must be 4672 for Lc0-style move representation.")

# --- Lc0 Move Encoding Constants ---
# Based on common understanding of Lc0/AlphaZero action space for chess.
# There are 73 possible "move types" from any given square.
# 8 knight moves
# 56 queen moves (8 directions * 7 distances)
# 9 pawn underpromotions (3 directions [forward, capture left, capture right] * 3 pieces [N, B, R])
# Total = 8 + 56 + 9 = 73 types.
# 64 squares * 73 types/square = 4672 actions.

# Knight moves: (delta_rank, delta_file)
# These are offsets from the source square.
# Note: In chess, rank increases upwards (positive delta_rank), file increases to the right (positive delta_file)
KNIGHT_MOVES_DELTAS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
] # 8 knight moves

# Queen moves (sliding pieces): (delta_rank, delta_file) for unit vectors
QUEEN_MOVES_UNIT_DELTAS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),  # Rook directions (N, S, E, W)
    (1, 1), (1, -1), (-1, 1), (-1, -1) # Bishop directions (NE, NW, SE, SW)
] # 8 directions

# For queen moves, we also need distances 1 through 7.
# Total queen-like moves = 8 directions * 7 distances = 56.

# Pawn underpromotions (N, B, R)
# Queen promotions are handled by the "queen moves" part of the policy if a pawn moves like a queen.
# This section is specifically for N, B, R promotions.
# Directions for pawn promotion (from pawn's perspective, assuming white):
#   - Forward (0 file change)
#   - Capture right (+1 file change)
#   - Capture left (-1 file change)
# Promotion pieces: Knight, Bishop, Rook (Queen promotion is a normal queen-like move)
PROMOTION_PIECES_ORDER = [chess.KNIGHT, chess.BISHOP, chess.ROOK] # 3 pieces

# Action mapping will be:
# Index = from_square * 73 + move_type_index_within_square
# move_type_index_within_square:
#   0-7: Knight moves
#   8-63: Queen moves (8 directions * 7 distances)
#   64-72: Underpromotions (3 directions * 3 pieces)

_ACTION_MAP_INITIALIZED = False # Not needed anymore as mapping is now algorithmic

# --- Algorithmic Move Mapping Functions ---

def get_move_plane(move: chess.Move) -> int:
    """
    Determines the 'move_type_index_within_square' (0-72) for a given chess.Move.
    This function assumes the move is from the perspective of the current player
    (i.e., if it's Black to move, the move object is for Black).
    The board representation should be canonical (current player is "White" on plane 0-5).
    """
    from_sq = move.from_square
    to_sq = move.to_square

    delta_rank = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    delta_file = chess.square_file(to_sq) - chess.square_file(from_sq)

    # 1. Pawn Underpromotions (N, B, R)
    if move.promotion and move.promotion in PROMOTION_PIECES_ORDER:
        # This is an underpromotion. Pawns promote on the 8th rank (rank index 7).
        # We need to map (delta_file) and promotion piece to an index from 0-8 for underpromotions.
        # delta_file: -1 (capture left), 0 (forward), 1 (capture right)
        # Mapped delta_file_idx: 0 for -1, 1 for 0, 2 for 1.
        df_map = {-1: 0, 0: 1, 1: 2}
        if delta_file not in df_map: return -1 # Should not happen for pawn promotion
        
        file_dir_idx = df_map[delta_file] # 0, 1, or 2

        try:
            promo_piece_idx = PROMOTION_PIECES_ORDER.index(move.promotion) # 0 for N, 1 for B, 2 for R
        except ValueError:
            return -1 # Should not happen if move.promotion is in PROMOTION_PIECES_ORDER

        # Index for underpromotions: 64 + file_dir_idx * 3 + promo_piece_idx
        return 64 + file_dir_idx * len(PROMOTION_PIECES_ORDER) + promo_piece_idx

    # 2. Knight Moves
    # Check if (delta_rank, delta_file) matches any knight move delta
    try:
        knight_move_idx = KNIGHT_MOVES_DELTAS.index((delta_rank, delta_file))
        # Check if the moving piece is actually a knight
        # This check isn't strictly needed if we assume the NN uses this mapping correctly,
        # but good for robust conversion from a chess.Move object.
        # board = chess.Board() # Need a board context if we want to check piece type
        # board.set_piece_at(from_sq, chess.Piece(chess.KNIGHT, chess.WHITE)) # Dummy piece
        # if board.piece_at(from_sq).piece_type == chess.KNIGHT:
        return knight_move_idx # 0-7
    except ValueError:
        pass # Not a knight move delta

    # 3. Queen-like Moves (including Queen promotions)
    # These are sliding moves (Rook or Bishop directions)
    # Find the unit direction vector and the distance.
    unit_dr, unit_df = 0, 0
    distance = 0

    if delta_rank == 0 and delta_file != 0: # Horizontal
        unit_df = 1 if delta_file > 0 else -1
        distance = abs(delta_file)
    elif delta_file == 0 and delta_rank != 0: # Vertical
        unit_dr = 1 if delta_rank > 0 else -1
        distance = abs(delta_rank)
    elif abs(delta_rank) == abs(delta_file) and delta_rank != 0: # Diagonal
        unit_dr = 1 if delta_rank > 0 else -1
        unit_df = 1 if delta_file > 0 else -1
        distance = abs(delta_rank)
    else:
        # Not a knight, underpromotion, or queen-like move by deltas.
        # This could be an error or a move type not covered (e.g. castling if treated specially)
        # Castling is typically represented as a king move of 2 squares.
        return -1 # Indicates move doesn't fit the 73 types cleanly

    if not (1 <= distance <= 7):
        return -1 # Distance out of bounds for queen moves

    try:
        direction_idx = QUEEN_MOVES_UNIT_DELTAS.index((unit_dr, unit_df)) # 0-7
    except ValueError:
        return -1 # Should not happen if logic above is correct

    # Index for queen moves: 8 (offset for knight moves) + direction_idx * 7 (max_distance) + (distance - 1)
    return 8 + direction_idx * 7 + (distance - 1)


def move_to_action_idx(move: chess.Move, board_turn_was_white: bool) -> int:
    """
    Converts a chess.Move object to its corresponding AlphaZero action index (0-4671).
    `board_turn_was_white`: True if it was White's turn when this move was made/considered,
                           False if it was Black's turn. This is crucial for handling
                           the `from_square` correctly in the canonical representation.
    """
    from_sq = move.from_square
    
    # If it was Black's turn, the board state given to the NN was flipped.
    # So, the `from_square` from the move (which is absolute 0-63) needs to be
    # flipped to match the NN's perspective.
    # chess.square_mirror(sq) flips the rank.
    canonical_from_sq = chess.square_mirror(from_sq) if not board_turn_was_white else from_sq

    move_plane = get_move_plane(move)
    if move_plane == -1:
        # This move doesn't map to one of the 73 defined types.
        # Could be castling if not handled as king move, or an error.
        # Or a pawn move that's not a promotion.
        # For pawn non-promotion moves, they should fall under "Queen-like moves" (distance 1 or 2, vertical)
        # Make sure `get_move_plane` correctly identifies pawn pushes/captures as queen-like moves.
        # Example: e2e4 (from White's view): from_sq=E2, to_sq=E4. delta_rank=2, delta_file=0.
        # This is QUEEN_MOVES_UNIT_DELTAS index 0 (1,0), distance 2.
        # Plane = 8 + 0*7 + (2-1) = 9.
        #
        # Example: e7e5 (Black's move). from_sq=E7, to_sq=E5.
        #   If board_turn_was_white=False (it was Black's turn):
        #     canonical_from_sq = mirror(E7) = E2.
        #     The move object `move` is still e7e5.
        #     get_move_plane(chess.Move.from_uci("e7e5")) ->
        #       delta_rank = rank(5)-rank(7) = -2. delta_file=0.
        #       This is unit_dr=-1, unit_df=0 (idx 1 in QUEEN_MOVES_UNIT_DELTAS), distance 2.
        #       Plane = 8 + 1*7 + (2-1) = 16.
        # This needs to be consistent. The `move` object given to `get_move_plane` should
        # reflect the canonical perspective if `canonical_from_sq` is used.
        #
        # Let's refine: get_move_plane should operate on deltas relative to the canonical board.
        # So, if it was Black's turn, the move itself needs to be "flipped" before getting deltas.
        
        canonical_to_sq = chess.square_mirror(move.to_square) if not board_turn_was_white else move.to_square
        
        # Create a "canonical move" if it was black's turn for delta calculation
        # Promotion piece also needs to be considered from the canonical perspective (no change needed for piece type)
        if not board_turn_was_white:
            # Create a new move object representing the move on the flipped board
            # This is tricky because the `move.promotion` is already correct.
            # The deltas should be calculated as if the move was made by White on the canonical board.
            # from_sq_for_delta = canonical_from_sq
            # to_sq_for_delta = canonical_to_sq
            # delta_rank = chess.square_rank(to_sq_for_delta) - chess.square_rank(from_sq_for_delta)
            # delta_file = chess.square_file(to_sq_for_delta) - chess.square_file(from_sq_for_delta)
            # The `get_move_plane` as written uses absolute deltas from the original move.
            # This should be fine if `canonical_from_sq` is used for the `from_square_idx * 73` part.
            # The "type" of move (knight, queen-like direction) is invariant to board flipping.
            pass # `move_plane = get_move_plane(move)` should work on the original move's structure.

        if move_plane == -1: # Re-check after potential clarification
            # print(f"Warning (move_to_action_idx): Move {move.uci()} from sq {chess.square_name(from_sq)} "
            #       f"(canonical_from_sq {chess.square_name(canonical_from_sq)}) "
            #       f"did not map to a valid move plane. board_turn_was_white={board_turn_was_white}")
            return -1 
            
    action_idx = canonical_from_sq * 73 + move_plane
    
    if not (0 <= action_idx < ACTION_SIZE):
        # print(f"Warning (move_to_action_idx): Calculated action_idx {action_idx} is out of bounds.")
        return -1
        
    return action_idx


def action_idx_to_move(action_idx: int, board: chess.Board) -> chess.Move | None:
    """
    Converts an action index (0-4671) back to a chess.Move object.
    The `board` is crucial because:
    1. It tells us whose turn it is (for handling the canonical_from_sq).
    2. It's needed to check if the generated move is legal.
    3. It tells us the piece type on the from_square (needed for pawn promotions).
    """
    if not (0 <= action_idx < ACTION_SIZE):
        # print(f"Warning (action_idx_to_move): action_idx {action_idx} is out of bounds.")
        return None

    original_from_sq_idx = action_idx // 73
    move_plane = action_idx % 73

    # Determine the actual `from_square` on the real board.
    # If it's Black's turn, the `original_from_sq_idx` was from a canonical (White's view) board.
    # So, we need to mirror it back to Black's perspective.
    current_player_from_sq = chess.square_mirror(original_from_sq_idx) if board.turn == chess.BLACK else original_from_sq_idx
    
    piece_on_from_sq = board.piece_at(current_player_from_sq)
    if piece_on_from_sq is None:
        # This can happen if the NN suggests a move from an empty square.
        # print(f"Warning (action_idx_to_move): No piece on calculated from_square {chess.square_name(current_player_from_sq)} for action_idx {action_idx} on board:\n{board}")
        return None

    # --- Decode move_plane (0-72) back to a move ---
    
    # 1. Knight Moves (plane 0-7)
    if 0 <= move_plane <= 7:
        if piece_on_from_sq.piece_type != chess.KNIGHT: return None # Piece mismatch
        delta_rank, delta_file = KNIGHT_MOVES_DELTAS[move_plane]
        
        # Calculate to_square based on current_player_from_sq and deltas
        # Remember: deltas are relative to "player's forward"
        # If black to move, a positive delta_rank means moving towards rank 1.
        # The KNIGHT_MOVE_DELTAS are defined as if player is moving "up" the board.
        # This needs to be adjusted if it's black's turn.
        # Or, more simply, the deltas define the target square from source square based on global board coordinates.
        # Let's assume deltas are global: +rank is towards rank 8, +file towards h-file.
        
        target_rank = chess.square_rank(current_player_from_sq) + delta_rank
        target_file = chess.square_file(current_player_from_sq) + delta_file

        if not (0 <= target_rank <= 7 and 0 <= target_file <= 7): return None # Off board
        to_sq = chess.square(target_file, target_rank)
        move = chess.Move(current_player_from_sq, to_sq)

    # 2. Queen-like Moves (plane 8-63)
    elif 8 <= move_plane <= 63:
        # Piece could be P, B, R, Q, K (K for castling, P for push/capture/Q-promotion)
        plane_offset = move_plane - 8
        direction_idx = plane_offset // 7
        distance = (plane_offset % 7) + 1
        
        unit_dr, unit_df = QUEEN_MOVES_UNIT_DELTAS[direction_idx]

        target_rank = chess.square_rank(current_player_from_sq) + unit_dr * distance
        target_file = chess.square_file(current_player_from_sq) + unit_df * distance

        if not (0 <= target_rank <= 7 and 0 <= target_file <= 7): return None # Off board
        to_sq = chess.square(target_file, target_rank)
        
        # Handle Queen promotion for pawns
        promotion_piece = None
        if piece_on_from_sq.piece_type == chess.PAWN:
            # White promotes on rank 7 (to rank 8), Black on rank 0 (to rank 1)
            is_white_pawn_promo = (board.turn == chess.WHITE and chess.square_rank(current_player_from_sq) == 6 and target_rank == 7)
            is_black_pawn_promo = (board.turn == chess.BLACK and chess.square_rank(current_player_from_sq) == 1 and target_rank == 0)
            if is_white_pawn_promo or is_black_pawn_promo:
                promotion_piece = chess.QUEEN # This part of action space is for Q-promo or non-promo queen-like moves

        move = chess.Move(current_player_from_sq, to_sq, promotion=promotion_piece)

    # 3. Underpromotions (plane 64-72)
    elif 64 <= move_plane <= 72:
        if piece_on_from_sq.piece_type != chess.PAWN: return None # Must be a pawn
        
        plane_offset = move_plane - 64
        file_dir_idx = plane_offset // len(PROMOTION_PIECES_ORDER) # 0, 1, or 2
        promo_piece_idx = plane_offset % len(PROMOTION_PIECES_ORDER) # 0, 1, or 2

        promo_piece_type = PROMOTION_PIECES_ORDER[promo_piece_idx]

        # Map file_dir_idx back to delta_file: 0 -> -1, 1 -> 0, 2 -> +1
        df_map_inv = {0: -1, 1: 0, 2: 1}
        delta_file = df_map_inv[file_dir_idx]

        # Determine target rank (always next rank for pawn promotion)
        delta_rank = 1 if board.turn == chess.WHITE else -1 
        
        target_rank = chess.square_rank(current_player_from_sq) + delta_rank
        target_file = chess.square_file(current_player_from_sq) + delta_file
        
        # Check if pawn is actually on its pre-promotion rank
        is_white_pre_promo_rank = (board.turn == chess.WHITE and chess.square_rank(current_player_from_sq) == 6)
        is_black_pre_promo_rank = (board.turn == chess.BLACK and chess.square_rank(current_player_from_sq) == 1)
        if not (is_white_pre_promo_rank or is_black_pre_promo_rank): return None # Not on correct rank to promote

        if not (0 <= target_rank <= 7 and 0 <= target_file <= 7): return None # Off board
        to_sq = chess.square(target_file, target_rank)
        move = chess.Move(current_player_from_sq, to_sq, promotion=promo_piece_type)
        
    else: # Should not happen if move_plane is 0-72
        return None

    # Final check: is the generated move legal on the board?
    if move in board.legal_moves:
        return move
    else:
        # This is common if NN policy is noisy. MCTS selection should filter using legal_mask.
        # print(f"Debug (action_idx_to_move): Generated move {move.uci()} for idx {action_idx} is NOT legal. Board:\n{board}")
        # print(f"   Original from_sq_idx: {original_from_sq_idx}, move_plane: {move_plane}")
        # print(f"   current_player_from_sq: {chess.square_name(current_player_from_sq)}, piece: {piece_on_from_sq}")
        # print(f"   Legal moves: {[m.uci() for m in board.legal_moves]}")
        return None


# --- Original utility functions, adapted to use new move mapping logic ---

def board_to_tensor(board: chess.Board):
    # (Implementation from your provided code - seems mostly fine, check repetition/castling planes carefully)
    # Ensure INPUT_SHAPE is (channels, height, width), e.g., (18, 8, 8)
    tensor = np.zeros(INPUT_SHAPE, dtype=np.float32)
    piece_to_plane = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }

    for r_idx in range(8): # rank index
        for f_idx in range(8): # file index
            sq = chess.square(f_idx, r_idx)
            piece = board.piece_at(sq)
            if piece:
                plane_idx = piece_to_plane[(piece.piece_type, piece.color)]
                tensor[plane_idx, r_idx, f_idx] = 1.0


    if board.is_repetition(2): tensor[13, :, :] = 1.0
    elif board.is_repetition(1): tensor[12, :, :] = 1.0
    
    if board.has_kingside_castling_rights(chess.WHITE): tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[15, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[16, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[17, :, :] = 1.0
    
    if board.turn == chess.BLACK:
        canonical_tensor = np.copy(tensor)
        white_pieces = np.copy(canonical_tensor[0:6, :, :])
        black_pieces = np.copy(canonical_tensor[6:12, :, :])
        canonical_tensor[0:6, :, :] = black_pieces
        canonical_tensor[6:12, :, :] = white_pieces
        
        white_castling = np.copy(canonical_tensor[14:16, :, :])
        black_castling = np.copy(canonical_tensor[16:18, :, :])
        canonical_tensor[14:16, :, :] = black_castling
        canonical_tensor[16:18, :, :] = white_castling
        
        for i in range(canonical_tensor.shape[0]):
             canonical_tensor[i, :, :] = np.flip(canonical_tensor[i, :, :], axis=0)
        tensor = canonical_tensor

    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)


def get_policy_target(board: chess.Board, mcts_policy_dict: dict, turn_at_state: chess.Color):
    # mcts_policy_dict: {move_uci: probability_float}
    policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
    
    board_turn_was_white = (turn_at_state == chess.WHITE)

    for move_uci, prob in mcts_policy_dict.items():
        try:
            move_obj = board.parse_uci(move_uci) # Use board to parse, ensures it's valid for context
            if move_obj not in board.legal_moves: # Extra safety: MCTS should only give legal moves
                # print(f"Warning (get_policy_target): MCTS move {move_uci} not legal on board.")
                continue
        except ValueError:
            # print(f"Warning (get_policy_target): MCTS produced invalid UCI {move_uci} for board state.")
            continue

        action_idx = move_to_action_idx(move_obj, board_turn_was_white)
        
        if action_idx != -1:
            policy_target[action_idx] = prob
        # else:
            # print(f"Warning (get_policy_target): Failed to map MCTS move {move_uci} to action_idx.")

    s = np.sum(policy_target)
    if s > 1e-6:
        policy_target /= s
    elif mcts_policy_dict and board.legal_moves: # MCTS had moves, but mapping failed
        # print("CRITICAL Warning (get_policy_target): Policy target sum is zero. Mapping MCTS moves failed.")
        # Fallback: uniform over legal moves that can be mapped
        temp_legal_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        num_legal_mapped = 0
        for legal_move in board.legal_moves:
            idx = move_to_action_idx(legal_move, board_turn_was_white)
            if idx != -1:
                temp_legal_probs[idx] = 1.0
                num_legal_mapped += 1
        if num_legal_mapped > 0:
            policy_target = temp_legal_probs / num_legal_mapped
        
    return torch.tensor(policy_target, dtype=torch.float32)


def get_legal_actions_mask(board: chess.Board):
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    board_turn_was_white = (board.turn == chess.WHITE)
    for move in board.legal_moves:
        action_idx = move_to_action_idx(move, board_turn_was_white)
        if action_idx != -1: # If the legal move can be mapped
            mask[action_idx] = True
        # else:
            # print(f"Warning (get_legal_actions_mask): Legal move {move.uci()} could not be mapped to an action_idx.")
    return mask


if __name__ == '__main__':
    print("Testing Lc0-style move representation in utils.py...")

    # Test specific move to action index and back
    board = chess.Board() # White to move

    # Test e2e4
    move_e2e4 = chess.Move.from_uci("e2e4")
    idx_e2e4 = move_to_action_idx(move_e2e4, board_turn_was_white=True)
    print(f"e2e4: UCI='{move_e2e4.uci()}', Action Index={idx_e2e4}")
    if idx_e2e4 != -1:
        retrieved_e2e4 = action_idx_to_move(idx_e2e4, board)
        print(f"  Index {idx_e2e4} -> Move: {retrieved_e2e4.uci() if retrieved_e2e4 else 'None'}")
        assert retrieved_e2e4 == move_e2e4, "e2e4 mapping failed"

    # Test Knight move g1f3
    move_g1f3 = chess.Move.from_uci("g1f3")
    idx_g1f3 = move_to_action_idx(move_g1f3, board_turn_was_white=True)
    print(f"g1f3: UCI='{move_g1f3.uci()}', Action Index={idx_g1f3}")
    if idx_g1f3 != -1:
        retrieved_g1f3 = action_idx_to_move(idx_g1f3, board)
        print(f"  Index {idx_g1f3} -> Move: {retrieved_g1f3.uci() if retrieved_g1f3 else 'None'}")
        assert retrieved_g1f3 == move_g1f3, "g1f3 mapping failed"

    # Test a promotion move e7e8q (needs board context)
    board_promo = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1") # White pawn on e7
    move_e7e8q = chess.Move.from_uci("e7e8q")
    idx_e7e8q = move_to_action_idx(move_e7e8q, board_turn_was_white=True) # True, as it's White's move on board_promo
    print(f"e7e8q: UCI='{move_e7e8q.uci()}', Action Index={idx_e7e8q}")
    if idx_e7e8q != -1:
        retrieved_e7e8q = action_idx_to_move(idx_e7e8q, board_promo)
        print(f"  Index {idx_e7e8q} -> Move: {retrieved_e7e8q.uci() if retrieved_e7e8q else 'None'}")
        assert retrieved_e7e8q == move_e7e8q, "e7e8q mapping failed"

    # Test an underpromotion e7e8n
    move_e7e8n = chess.Move.from_uci("e7e8n")
    idx_e7e8n = move_to_action_idx(move_e7e8n, board_turn_was_white=True)
    print(f"e7e8n: UCI='{move_e7e8n.uci()}', Action Index={idx_e7e8n}")
    if idx_e7e8n != -1:
        retrieved_e7e8n = action_idx_to_move(idx_e7e8n, board_promo)
        print(f"  Index {idx_e7e8n} -> Move: {retrieved_e7e8n.uci() if retrieved_e7e8n else 'None'}")
        assert retrieved_e7e8n == move_e7e8n, "e7e8n mapping failed"
    
    # Test Black's move after White moves (for canonical from_square)
    board_black_turn = chess.Board()
    board_black_turn.push_uci("e2e4") # White moves
    # Now it's Black's turn
    move_e7e5_black = chess.Move.from_uci("e7e5")
    idx_e7e5_black = move_to_action_idx(move_e7e5_black, board_turn_was_white=False) # False, Black's turn
    print(f"Black e7e5: UCI='{move_e7e5_black.uci()}', Action Index={idx_e7e5_black}")
    if idx_e7e5_black != -1:
        # To test action_idx_to_move, the board must be in the state where Black is to move
        retrieved_e7e5_black = action_idx_to_move(idx_e7e5_black, board_black_turn)
        print(f"  Index {idx_e7e5_black} -> Move: {retrieved_e7e5_black.uci() if retrieved_e7e5_black else 'None'}")
        assert retrieved_e7e5_black == move_e7e5_black, "Black e7e5 mapping failed"
        # Check if canonical mapping works: Black e7e5 should map to the same type of action
        # from a mirrored square as White e2e4.
        # E7 (black) mirrored is E2. E5 (black) mirrored is E4.
        # move_plane for e7e5 (delta_rank=-2) should be different from e2e4 (delta_rank=2) if handled naively by get_move_plane
        # This highlights complexity. `get_move_plane` itself doesn't need to know turn, it works on move geometry.
        # `move_to_action_idx` handles canonical from_square.
        assert idx_e7e5_black != idx_e2e4 # They are different moves from different (canonical) squares
        # A more detailed check:
        # from_sq_e2e4 = chess.E2 -> 12
        # from_sq_e7e5_canonical = chess.square_mirror(chess.E7) = chess.E2 -> 12
        # move_plane_e2e4 = get_move_plane(move_e2e4)
        # move_plane_e7e5 = get_move_plane(move_e7e5_black)
        # assert move_plane_e2e4 == move_plane_e7e5 # This should be true IF pawn pushes are treated identically regardless of color.
                                                 # (1,0) distance 2 vs (-1,0) distance 2. Depends on QUEEN_MOVES_UNIT_DELTAS order.
                                                 # (1,0) is index 0. (-1,0) is index 1. So planes would be different. This is correct.
        # So idx_e2e4 = 12*73 + plane_e2e4.  idx_e7e5_black = 12*73 + plane_e7e5.

    # Test legal actions mask
    mask = get_legal_actions_mask(board)
    print(f"\nInitial board legal mask: {torch.sum(mask).item()} trues out of {ACTION_SIZE}")
    assert torch.sum(mask).item() == len(list(board.legal_moves)), "Mask true count mismatch with legal moves"
    
    # Test a few more complex positions and moves if possible.
    print("\nUtils.py Lc0-style move tests finished. Further manual verification recommended.")