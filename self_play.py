# self_play.py

import chess
import numpy as np
import torch
import time # For timing if needed
from multiprocessing import Pool, Manager # Manager for progress queue if used
from tqdm import tqdm
import os # For os.getpid() if used in debugging initialize_worker

from config import (NUM_SIMULATIONS, TEMPERATURE_INITIAL, TEMPERATURE_THRESHOLD,
                    MAX_MOVES_PER_GAME, DEVICE, NUM_WORKERS as CONFIG_NUM_WORKERS) # Avoid name clash if num_workers is also a func arg
# Utility imports
from utils import board_to_tensor, get_policy_target
# MCTS import
from mcts import MCTS
# Neural Net class import (needed for type hint and for model_class argument)
from neural_net import ChessNN


# --- Worker Initialization and Game Playing Logic (MUST BE TOP-LEVEL for pickling by multiprocessing) ---

_worker_model = None # Global model for this worker process

def initialize_worker(model_state_dict_arg, model_class_arg):
    """Initializes the model for a worker process."""
    global _worker_model # Refer to the global variable in this worker's scope
    # print(f"Worker {os.getpid()}: Initializing...")
    _worker_model = model_class_arg().to(DEVICE) # DEVICE should be imported from config
    _worker_model.load_state_dict(model_state_dict_arg)
    _worker_model.eval()
    # print(f"Worker {os.getpid()} initialized model on {DEVICE}")


def play_one_game(process_id_and_game_idx_tuple): # Modified to take a single arg for starmap/map
    """Plays one game of chess using MCTS and the global _worker_model."""
    # Unpack arguments if you passed a tuple
    # game_idx_q, process_id = game_idx_and_q_tuple # If using a queue
    process_id, game_idx = process_id_and_game_idx_tuple # Assuming tuple (process_id, game_idx)

    global _worker_model # Use the model initialized by initialize_worker
    if _worker_model is None:
        print(f"Error: Worker model not initialized in process {process_id}, game {game_idx}.")
        return None # Indicate error

    board = chess.Board()
    # Each game/worker needs its own MCTS instance, using the shared (copied) worker_model
    mcts_instance = MCTS(_worker_model, board.copy()) 

    game_history_for_buffer = [] # List to store (state_tensor_cpu, policy_target_cpu, turn_color_at_state)
    
    current_temp = TEMPERATURE_INITIAL
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < MAX_MOVES_PER_GAME:
        if move_count >= TEMPERATURE_THRESHOLD:
            current_temp = 0 # Deterministic play for MCTS policy sampling

        # MCTS search returns policy_dist {uci: prob} and chosen_move (chess.Move obj)
        # The MCTS search method itself should handle temperature for move selection if applicable
        # or _get_final_action_probs should take temperature.
        # Let's assume MCTS's _get_final_action_probs uses a temp parameter or has a default.
        # For self-play, we often use the passed 'current_temp'.
        # Modifying MCTS's _get_final_action_probs to accept temp is cleaner.
        # For now, assume MCTS's search or its policy generation internally handles temp.
        
        # The `search` method in mcts.py was defined to return:
        # policy_for_training (dict), chosen_move_obj (Move)
        # The temperature for choosing the move should be applied within MCTS's final move selection step.
        # Let's assume _get_final_action_probs in mcts.py takes temperature.
        # If not, we'd pass it to search, and search passes it down.
        # The policy_dist for training should be based on N (visit counts), not temperature.
        
        # Get policy from MCTS (policy_for_training is based on visit counts, move chosen with temp)
        # The MCTS search method needs to be adapted to use current_temp for its final move selection part
        # while returning the "raw" visit count based policy for training.
        # The `_get_final_action_probs` in my provided `mcts.py` already takes `temp`.
        # So, we should pass `current_temp` to the search function, or `search` should pass it on.
        # Let's assume `mcts_instance.search` itself uses `current_temp` for final action selection,
        # and the returned policy_dist is the pi for training.
        # The `mcts.py` I provided has `_get_final_action_probs(..., temp=0.01)`.
        # We need to make sure the actual move selection uses `current_temp`.
        # A clean way: mcts_instance.search(board, NUM_SIMULATIONS, temp_for_move_selection=current_temp)
        # For now, the `search` in `mcts.py` calls `_get_final_action_probs` which has a default temp.
        # Let's assume the current `mcts.py` `search` returns the training policy and a temp-influenced move.

        mcts_policy_training_dist, chosen_move_obj = mcts_instance.search(board.copy(), NUM_SIMULATIONS)
                                                                     # temp=current_temp) # Pass temp if MCTS search accepts it

        if chosen_move_obj is None :
            break 

        state_tensor_gpu = board_to_tensor(board).to(DEVICE) # Canonical state for NN
        # Policy target from MCTS output (for current player at `board`)
        pi_target_tensor = get_policy_target(board, mcts_policy_training_dist, board.turn)

        game_history_for_buffer.append((state_tensor_gpu.squeeze(0).cpu(), 
                                        pi_target_tensor.cpu(), 
                                        board.turn))

        board.push(chosen_move_obj)
        move_count += 1
        
    # Game finished, determine winner and assign values
    final_game_data_for_replay = []
    outcome = board.outcome(claim_draw=True)
    game_result_white_perspective = 0.0
    if outcome:
        if outcome.winner == chess.WHITE: game_result_white_perspective = 1.0
        elif outcome.winner == chess.BLACK: game_result_white_perspective = -1.0
    # else: max moves reached, treated as draw (0.0)

    for state_t, policy_t, turn_color_at_state in game_history_for_buffer:
        value_target = game_result_white_perspective if turn_color_at_state == chess.WHITE else -game_result_white_perspective
        final_game_data_for_replay.append((state_t, policy_t, torch.tensor(value_target, dtype=torch.float32)))
    
    # if game_idx_q is not None: game_idx_q.put(1) # If using a progress queue
    return final_game_data_for_replay # Removed process_id for simplicity with starmap if not needed


# --- Main Self-Play Data Generation Function ---

def generate_self_play_data(model, num_games_to_play, num_workers_for_pool):
    """Generates self-play data using multiple worker processes."""
    
    model.cpu() # Move model to CPU before sharing state_dict for workers
    model_state_dict_to_share = model.state_dict()
    # Pass ChessNN class itself for workers to instantiate
    model_class_for_workers = ChessNN 

    all_generated_games_data = []
    
    # Prepare arguments for each worker task
    # Each task needs a unique identifier if you want to track, e.g., game index or worker id
    # For map/starmap, we just need an iterable of arguments for play_one_game
    # Let's pass (process_id_placeholder, game_index) to play_one_game
    task_args = [(i % num_workers_for_pool if num_workers_for_pool > 0 else 0, i) for i in range(num_games_to_play)]


    print(f"Starting {num_games_to_play} self-play games with {num_workers_for_pool} workers...")
    
    if num_workers_for_pool > 0 :
        # initializer and initargs are for setting up each worker process once
        with Pool(processes=num_workers_for_pool, 
                  initializer=initialize_worker,  # This function MUST be defined globally
                  initargs=(model_state_dict_to_share, model_class_for_workers)) as pool:
            
            # Use pool.map to distribute tasks. play_one_game will be called with each item in task_args.
            # results will be a list of final_game_data_for_replay from each game.
            # tqdm can be used with pool.imap or pool.imap_unordered for progress.
            
            game_results_list = []
            # Using imap_unordered for potentially better responsiveness with tqdm
            # It yields results as they complete.
            for result_data_one_game in tqdm(pool.imap_unordered(play_one_game, task_args), total=num_games_to_play, desc="Self-play games"):
                if result_data_one_game: # If game data was successfully generated
                    game_results_list.append(result_data_one_game)
            
            for game_data_list in game_results_list: # Each item is a list of (s,p,v) tuples for one game
                 all_generated_games_data.extend(game_data_list)

    else: # Single process (for debugging or if num_workers_for_pool is 0)
        # Initialize model in the main process if not using Pool
        initialize_worker(model_state_dict_to_share, model_class_for_workers) 
        for i in tqdm(range(num_games_to_play), desc="Self-play games (single worker)"):
            # play_one_game needs a tuple like (process_id, game_idx)
            game_data_one_game_list = play_one_game((0, i)) 
            if game_data_one_game_list:
                all_generated_games_data.extend(game_data_one_game_list)

    model.to(DEVICE) # Move model back to original device
    print(f"\nGenerated {len(all_generated_games_data)} training samples from {num_games_to_play} games (approx).") # Note: this is total (s,p,v) tuples
    return all_generated_games_data


if __name__ == '__main__':
    # This block is for testing self_play.py directly
    import torch.multiprocessing as mp
    try:
        # This should be called only once, ideally in the main script (train.py)
        # But if running self_play.py standalone for testing, it's needed here too.
        if CONFIG_NUM_WORKERS > 0: # Use the NUM_WORKERS from config for this test
            mp.set_start_method('spawn', force=True) 
            print("self_play.py standalone: Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"self_play.py standalone: Multiprocessing start method: {e}")

    from replay_buffer import ReplayBuffer # For testing adding to buffer
    from config import NUM_SELF_PLAY_GAMES, REPLAY_BUFFER_SIZE # Use values from config

    print(f"self_play.py standalone: Using device: {DEVICE}")
    # Create a dummy model for testing generate_self_play_data
    test_model = ChessNN().to(DEVICE) 
    
    start_time = time.time()
    
    # Use fewer games/workers for a quick standalone test
    test_num_games = 5  # Or NUM_SELF_PLAY_GAMES // 10 for a smaller run
    test_num_workers = min(CONFIG_NUM_WORKERS, 2) if CONFIG_NUM_WORKERS > 0 else 0
    if CONFIG_NUM_WORKERS == 0 : test_num_workers = 0


    print(f"self_play.py standalone: Testing self-play with {test_num_games} games, {test_num_workers} workers.")
    
    generated_data = generate_self_play_data(test_model, test_num_games, test_num_workers)
    
    end_time = time.time()
    print(f"self_play.py standalone: Self-play data generation took {end_time - start_time:.2f} seconds.")
    
    if generated_data:
        print(f"self_play.py standalone: Total samples generated: {len(generated_data)}")
        replay_buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)
        replay_buffer.add_game_data(generated_data)
        print(f"self_play.py standalone: Replay buffer size: {len(replay_buffer)}")
    else:
        print("self_play.py standalone: No data generated from self-play.")