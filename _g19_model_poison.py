import torch
import torch.nn as nn
import copy

# 1. Define your model class (REPLACE with your ACTUAL model)
class MyModel(nn.Module):
    def __init__(self, input_size):  # Add input size as parameter
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 2. Function to poison the model weights
def poison_model(model_path, poisoning_percentage=0.1):
    try:
        state_dict = torch.load(model_path, weights_only=True)
        input_size = state_dict['fc1.weight'].shape[1]
        model = MyModel(input_size)
        model.load_state_dict(state_dict)  # Load the original weights
        print(f"Model loaded successfully from: {model_path}")

    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None  # Return None if loading fails

    poisoned_state_dict = copy.deepcopy(state_dict) # Deep copy the state dict

    for key, tensor in poisoned_state_dict.items():
        if 'weight' in key: # Only poison weights, not biases
            num_elements = tensor.numel()
            num_poisoned = int(num_elements * poisoning_percentage)
            indices = torch.randperm(num_elements)[:num_poisoned] # Random indices

            # Example poisoning: Add small random noise to weights
            noise = torch.randn_like(tensor.view(-1)[indices]) * 0.01  # Adjust noise level
            tensor.view(-1)[indices] += noise

            print(f"Poisoned {num_poisoned} elements in {key}")

    # Save the poisoned model
    poisoned_model_path = "poisoned_" + model_path  # New filename
    torch.save(poisoned_state_dict, poisoned_model_path)
    print(f"Poisoned model saved to: {poisoned_model_path}")

    return poisoned_model_path # Return path to the poisoned model

# 3. Example usage
if __name__ == "__main__":
    model_path = "model.pth"  # Replace with your .pth file
    poisoned_model_path = poison_model(model_path, poisoning_percentage=0.05)  # 5% poisoning

    if poisoned_model_path:
        # You can now use the poisoned model (poisoned_model_path) for testing or further analysis
        print("Poisoning complete. You can now use the poisoned model at:", poisoned_model_path)
    else:
        print("Model poisoning failed.")
