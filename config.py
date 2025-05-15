import torch

# --- General ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAME_NAME = "chess"
SEED = 42

# --- Neural Network ---
INPUT_SHAPE = (18, 8, 8)
ACTION_SIZE = 4672
RESIDUAL_BLOCKS = 7
CONV_FILTERS = 128

# --- MCTS ---
NUM_SIMULATIONS = 100 #100
CPUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# --- Self-Play ---
NUM_SELF_PLAY_GAMES = 500 # 500
NUM_WORKERS = 8 # Adjust to CPU cores
TEMPERATURE_INITIAL = 1.0
TEMPERATURE_THRESHOLD = 30 #30
MAX_MOVES_PER_GAME = 250   

# --- Training ---
REPLAY_BUFFER_SIZE = 50000 #50000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS_PER_ITERATION = 2 #5
CHECKPOINT_INTERVAL = 1
TRAIN_SAMPLE_FRACTION = 0.8

# --- Evaluation (Optional) ---
EVAL_GAMES = 20
EVAL_WIN_RATE = 0.55