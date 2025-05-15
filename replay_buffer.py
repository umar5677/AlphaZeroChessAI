from collections import deque
import random
from config import REPLAY_BUFFER_SIZE, SEED # These are fine

random.seed(SEED)

class ReplayBuffer:
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add_game_data(self, game_data):
        # game_data is a list of tuples: [(state_tensor_cpu, policy_target_cpu, value_target_cpu), ...]
        # state_tensor should be already on CPU and detached.
        for experience in game_data:
            self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None # Not enough data to sample
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)

# The if __name__ == '__main__': block for testing can go here.
# Example:
if __name__ == '__main__':
    import torch # For dummy data
    # from config import INPUT_SHAPE, ACTION_SIZE # If needed for dummy data shape

    buffer = ReplayBuffer(max_size=100)
    
    game_history = []
    for _ in range(5):
        # Use fixed shapes for dummy data if not importing from config here
        dummy_state = torch.randn(18, 8, 8) 
        dummy_policy_target = torch.rand(4672)
        dummy_policy_target /= dummy_policy_target.sum()
        dummy_value_target = torch.tensor(random.choice([-1.0, 0.0, 1.0]), dtype=torch.float32)
        game_history.append((dummy_state, dummy_policy_target, dummy_value_target))
        
    buffer.add_game_data(game_history)
    print(f"Buffer size: {len(buffer)}")

    sample_batch = buffer.sample(batch_size=2)
    if sample_batch:
        states, policies, values = zip(*sample_batch)
        states_tensor = torch.stack(states)
        policies_tensor = torch.stack(policies)
        values_tensor = torch.stack(values).unsqueeze(1)

        print("Sampled States shape:", states_tensor.shape)
        print("Sampled Policies shape:", policies_tensor.shape)
        print("Sampled Values shape:", values_tensor.shape)
    else:
        print("Not enough data to sample.")