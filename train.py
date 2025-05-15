import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np
import random
import time
import torch.multiprocessing as mp # For setting start method

# Configuration imports
from config import (DEVICE, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS_PER_ITERATION,
                    NUM_SELF_PLAY_GAMES, NUM_WORKERS, REPLAY_BUFFER_SIZE, SEED,
                    CHECKPOINT_INTERVAL, GAME_NAME, TRAIN_SAMPLE_FRACTION, INPUT_SHAPE, ACTION_SIZE)

# Project module imports
from neural_net import ChessNN
from replay_buffer import ReplayBuffer
from self_play import generate_self_play_data
# Note: train.py itself might not directly need functions from utils.py,
# as self_play.py and mcts.py handle their own interactions with utils.py.
# If for some reason train.py needed a util function, it would be imported here.

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.type == 'cuda': # Check if CUDA is actually being used
    torch.cuda.manual_seed_all(SEED)


def train_nn(model, replay_buffer, optimizer, epochs, batch_size, sample_fraction):
    model.train() 
    total_policy_loss_accumulator = 0.0
    total_value_loss_accumulator = 0.0
    batches_processed_in_training = 0

    if len(replay_buffer) < batch_size:
        print(f"Warning: Not enough samples ({len(replay_buffer)}) in buffer for a full batch ({batch_size}). Skipping training for this iteration.")
        return 0.0, 0.0

    # --- DEFINE num_samples_to_draw_total HERE ---
    num_samples_to_draw_total = int(len(replay_buffer) * sample_fraction)
    if num_samples_to_draw_total < batch_size and len(replay_buffer) >= batch_size :
        num_samples_to_draw_total = len(replay_buffer) # Use all if fraction is too small but buffer is full enough for a batch
    elif num_samples_to_draw_total < batch_size: # Not enough samples even after considering full buffer for small fraction
        print(f"Warning: Calculated samples to draw ({num_samples_to_draw_total}) is less than batch_size ({batch_size}) even from buffer of size {len(replay_buffer)}. Skipping training.")
        return 0.0, 0.0
    # --- END OF DEFINITION ---

    print(f"Training on {num_samples_to_draw_total} samples from replay buffer (total size {len(replay_buffer)}).")

    epoch_training_data = None # Initialize to None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Sample data only once for all epochs in this training phase
        if epoch == 0: 
            epoch_training_data = replay_buffer.sample(num_samples_to_draw_total) 
            # Ensure sample didn't return None (e.g. if buffer became smaller concurrently, though unlikely here)
            if epoch_training_data is None or len(epoch_training_data) < batch_size:
                 print(f"Warning: Sampling {num_samples_to_draw_total} yielded too few samples ({len(epoch_training_data) if epoch_training_data else 0}). Skipping epoch.")
                 continue # Skip this epoch, might try again if epochs > 1 or break if issue persists

        if not epoch_training_data: # If sampling failed on epoch 0 or was None
            print("No training data available for this epoch. Skipping.")
            continue

        states_list, policies_list, values_list = zip(*epoch_training_data)
        
        # ... (rest of the try-except for tensor stacking, DataLoader creation, and inner training loop) ...
        # ... (as in the previous corrected version) ...
        try:
            # --- Create dataset with CPU tensors ---
            states_tensor = torch.stack(states_list)    # Keep on CPU
            policies_tensor = torch.stack(policies_list) # Keep on CPU
            values_tensor = torch.stack(values_list).unsqueeze(1) # Keep on CPU
        except Exception as e:
            print(f"Error during tensor stacking: {e}")
            print(f"  Data sample types: states[0]={type(states_list[0]) if states_list else 'N/A'}, policies[0]={type(policies_list[0]) if policies_list else 'N/A'}, values[0]={type(values_list[0]) if values_list else 'N/A'}")
            continue # Skip this epoch if stacking fails

        dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, 
                                  pin_memory=(DEVICE.type == 'cuda'))

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        
        for batch_idx, (s_batch_cpu, pi_batch_cpu, v_batch_cpu) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            s_batch = s_batch_cpu.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
            pi_batch = pi_batch_cpu.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
            v_batch = v_batch_cpu.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))

            optimizer.zero_grad()
            policy_logits, value_preds = model(s_batch)
            loss_pi = F.cross_entropy(policy_logits, pi_batch)
            loss_v = F.mse_loss(value_preds, v_batch)
            total_loss = loss_pi + loss_v
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += loss_pi.item()
            epoch_value_loss += loss_v.item()
            batches_processed_in_training += 1

        if len(train_loader) > 0:
             avg_epoch_pi_loss = epoch_policy_loss / len(train_loader)
             avg_epoch_v_loss = epoch_value_loss / len(train_loader)
             print(f"Epoch {epoch+1} Avg Loss -> Policy: {avg_epoch_pi_loss:.4f}, Value: {avg_epoch_v_loss:.4f}")
             total_policy_loss_accumulator += epoch_policy_loss
             total_value_loss_accumulator += epoch_value_loss
        else:
            print(f"Epoch {epoch+1} had no batches to process.")


    if batches_processed_in_training > 0:
        overall_avg_policy_loss = total_policy_loss_accumulator / batches_processed_in_training
        overall_avg_value_loss = total_value_loss_accumulator / batches_processed_in_training
        print(f"Overall Avg Training Batch Loss (across all epochs in this phase) -> Policy: {overall_avg_policy_loss:.4f}, Value: {overall_avg_value_loss:.4f}")
        return overall_avg_policy_loss, overall_avg_value_loss
    else:
        return 0.0, 0.0


    if batches_processed_in_training > 0:
        overall_avg_policy_loss = total_policy_loss_accumulator / batches_processed_in_training
        overall_avg_value_loss = total_value_loss_accumulator / batches_processed_in_training
        print(f"Overall Avg Training Batch Loss (across all epochs in this phase) -> Policy: {overall_avg_policy_loss:.4f}, Value: {overall_avg_value_loss:.4f}")
        return overall_avg_policy_loss, overall_avg_value_loss
    else:
        return 0.0, 0.0 # No training batches were processed


def main_training_loop(num_iterations=100):
    print(f"Starting AlphaZero training on {DEVICE} for {GAME_NAME}")
    # _initialize_action_maps() # This call is REMOVED as it's no longer needed from utils.py

    # --- Initialization ---
    current_model = ChessNN().to(DEVICE) # This is the model being trained
    optimizer = optim.AdamW(current_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    replay_buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)

    # Checkpoint loading
    iteration_start = 0
    checkpoint_dir = f"./checkpoints_{GAME_NAME}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")

    if os.path.exists(latest_checkpoint_path):
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
            current_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            iteration_start = checkpoint['iteration'] + 1
            # replay_buffer persistence is optional and not included here by default
            print(f"Resuming from iteration {iteration_start}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            iteration_start = 0 # Reset if checkpoint loading fails


    for iteration in range(iteration_start, num_iterations):
        print(f"\n--- Iteration {iteration}/{num_iterations} ---")
        
        # --- 1. Self-Play Phase ---
        print("Starting self-play phase...")
        start_time_self_play = time.time()
        
        # generate_self_play_data needs the current model's state_dict for workers
        # It's better if generate_self_play_data handles moving model to CPU internally if needed
        game_data = generate_self_play_data(current_model, NUM_SELF_PLAY_GAMES, NUM_WORKERS)
        
        if game_data: # Ensure game_data is not None or empty
            replay_buffer.add_game_data(game_data)
        else:
            print("Warning: No game data generated from self-play.")
            
        end_time_self_play = time.time()
        print(f"Self-play took {end_time_self_play - start_time_self_play:.2f}s. Buffer size: {len(replay_buffer)}")

        # --- 2. Training Phase ---
        # Minimum data threshold to start training (e.g., at least a few batches worth)
        min_data_for_training = BATCH_SIZE * 5 
        if len(replay_buffer) < min_data_for_training:
             print(f"Not enough data in replay buffer ({len(replay_buffer)} / {min_data_for_training}) to train. Skipping training for this iteration.")
        else:
            print("Starting training phase...")
            start_time_train = time.time()
            avg_policy_loss, avg_value_loss = train_nn(current_model, replay_buffer, optimizer, 
                                                       NUM_EPOCHS_PER_ITERATION, BATCH_SIZE,
                                                       TRAIN_SAMPLE_FRACTION)
            end_time_train = time.time()
            print(f"Training took {end_time_train - start_time_train:.2f}s.")
            if avg_policy_loss == 0.0 and avg_value_loss == 0.0 and iteration > iteration_start :
                print("Warning: Training resulted in zero loss, could indicate no data processed or other issues.")


        # --- 3. Save Checkpoint ---
        if (iteration % CHECKPOINT_INTERVAL == 0 and iteration >= iteration_start) or iteration == num_iterations - 1:
            checkpoint_path_iter = os.path.join(checkpoint_dir, f"model_iter_{iteration}.pth")
            try:
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': current_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path_iter)
                
                # Update latest_model.pth
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': current_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, latest_checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path_iter} and {latest_checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        
        # --- (Optional) 4. Evaluation Phase ---
        # (Skipped for brevity, but important for robust training)
        # Pit current_model against a previous best_model.
        # If current_model wins by a certain margin, it becomes the new best_model for self-play.
        # For now, self-play always uses current_model.

    print("\nTraining completed.")


if __name__ == '__main__':
    # IMPORTANT: Set multiprocessing start method for CUDA safety if using workers > 0
    # This should be done once at the beginning of the main script.
    if NUM_WORKERS > 0: # Only set if actually using multiprocessing
        try:
            # 'spawn' is generally safer for PyTorch with CUDA.
            # 'fork' (default on Linux) can cause issues with CUDA contexts in child processes.
            mp.set_start_method('spawn', force=True) 
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError as e:
            # This means it might have already been set, or we are in a context where it can't be changed.
            # Or, on some systems, 'spawn' might not be the default or preferred for other reasons.
            print(f"Note: Multiprocessing start method: {e} (may be already set or system default). Current method: {mp.get_start_method(allow_none=True)}")
            if mp.get_start_method(allow_none=True) != 'spawn':
                 print("Warning: Consider setting start_method to 'spawn' for CUDA compatibility if issues arise with workers.")
        
    # Set how many iterations you want to run in total for this training session
    # For a quick test, you might set this to a small number like 10.
    # For more serious training, it would be much larger (e.g., 100, 500, 1000+).
    total_training_iterations = 500  # Example: run for 10 iterations
    
    main_training_loop(num_iterations=total_training_iterations)