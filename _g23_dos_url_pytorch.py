import torch
import torch.nn as nn
import time
import os
import requests  # For responsiveness testing
import threading  # For concurrent testing

# 1. Define your model (REPLACE with your ACTUAL model - MUST MATCH)
class MyModel(nn.Module):  # Example - REPLACE with your actual model
    def __init__(self, input_size):
        super().__init__()
        # Example - MUST match your saved model's architecture
        self.fc1 = nn.Linear(input_size, 10)  # Example: Input size, Output 10
        self.fc2 = nn.Linear(10, 5)   # Example: Input 10, Output 5
        self.fc3 = nn.Linear(5, 2)    # Example: Input 5, Output 2

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 2. Function to simulate a denial-of-service attack (BLACK-BOX)
def dos_attack_blackbox(model_path, attack_duration=10):  # Duration in seconds
    try:
        state_dict = torch.load(model_path, weights_only=True)
        input_size = state_dict['fc1.weight'].shape[1]  # Get input size
        model = MyModel(input_size)  # Use DummyModel for loading
        model.load_state_dict(state_dict, strict=False)  # strict=False is important
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

# 3. Function to test responsiveness (separate thread)
def test_responsiveness(url):
    while True:  # Run continuously
        start_time = time.time()
        try:
            response = requests.get(url)  # Replace with your actual request method
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            latency = time.time() - start_time
            print(f"Responsiveness test: Latency = {latency:.4f} seconds")
        except requests.exceptions.RequestException as e:
            print(f"Responsiveness test: Error = {e}")
        time.sleep(1)  # Test every 1 second

# 4. Example Usage (BLACK-BOX VERSION - COMPLETE)
if __name__ == "__main__":
    model_path = "model.pth"  # Replace with the actual path to your model
    model_url = "http://your-model-url" # Replace with the URL of your model endpoint

    # --- VERY IMPORTANT: Inspect your model.pth and create matching DummyModel ---
    # Run this code ONCE to get the architecture:
    # import torch
    # state_dict = torch.load(model_path, weights_only=True)
    # for key, value in state_dict.items():
    #     print(key, value.shape)
    # --- Then, based on the output, create a DummyModel that matches the architecture.

    # Start responsiveness testing in a separate thread
    responsiveness_thread = threading.Thread(target=test_responsiveness, args=(model_url,))
    responsiveness_thread.daemon = True # Allow main thread to exit even if this thread is running
    responsiveness_thread.start()

    # Run the DoS attack
    dos_attack_blackbox(model_path, attack_duration=10)  # Attack for 10 seconds

    print("DoS test completed.")
