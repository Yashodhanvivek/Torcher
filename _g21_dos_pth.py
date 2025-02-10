import torch
import torch.nn as nn
import time
import os

# 1. Define your model (REPLACE with your ACTUAL model - MUST MATCH)
class MyModel(nn.Module):  # Example - REPLACE with your actual model
    def __init__(self, input_size):
        super().__init__()
        # Example - REPLACE with your actual layers and sizes
        self.fc1 = nn.Linear(input_size, 10)  # Example: Input 12, Output 10
        self.fc2 = nn.Linear(10, 5)   # Example: Input 10, Output 5
        self.fc3 = nn.Linear(5, 2)    # Example: Input 5, Output 2

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 2. Function to simulate a denial-of-service attack
def dos_attack(model_path, attack_duration=10):  # Duration in seconds
    try:
        state_dict = torch.load(model_path, weights_only=True)
        input_size = state_dict['fc1.weight'].shape[1]  # Get input size
        model = MyModel(input_size)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    start_time = time.time()
    while time.time() - start_time < attack_duration:
        # Generate a large batch of random input data
        batch_size = 1024  # Adjust batch size as needed
        input_data = torch.randn(batch_size, input_size)

        try:
            with torch.no_grad():  # Disable gradients for inference
                _ = model(input_data)  # Perform inference (discard output)
            print(f"Inference of batch size {batch_size} was successful")

        except Exception as e:
            print(f"Error during inference: {e}")
            # If you want to stop the attack on the first error, uncomment the line below:
            # break  # Stop the attack if an error occurs

        # Optional: Introduce a small delay to simulate more realistic attack patterns
        # time.sleep(0.01)  # Adjust delay as needed

    print(f"Denial-of-service attack finished after {attack_duration} seconds.")

# 3. Example Usage
if __name__ == "__main__":
    model_path = "model.pth"  # Replace with the actual path to your model
    dos_attack(model_path, attack_duration=5)  # Attack for 5 seconds
