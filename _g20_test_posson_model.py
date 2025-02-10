import torch
import torch.nn as nn
import os

# 1. Define your model class (REPLACE with your ACTUAL model - MUST MATCH)
class MyModel(nn.Module):
    def __init__(self, input_size):
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
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from: {model_path}")

    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

    poisoned_state_dict = copy.deepcopy(state_dict)

    for key, tensor in poisoned_state_dict.items():
        if 'weight' in key:
            num_elements = tensor.numel()
            num_poisoned = int(num_elements * poisoning_percentage)
            indices = torch.randperm(num_elements)[:num_poisoned]

            noise = torch.randn_like(tensor.view(-1)[indices]) * 0.01  # Adjust noise level
            tensor.view(-1)[indices] += noise

            print(f"Poisoned {num_poisoned} elements in {key}")

    poisoned_model_path = "poisoned_" + model_path
    torch.save(poisoned_state_dict, poisoned_model_path)
    print(f"Poisoned model saved to: {poisoned_model_path}")

    return poisoned_model_path

# 3. Function to test a model (clean or poisoned)
def test_model(model_path, test_data, test_target):
    try:
        state_dict = torch.load(model_path, weights_only=True)
        input_size = state_dict['fc1.weight'].shape[1] #Get input size
        model = MyModel(input_size)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            input_data = test_data[i].unsqueeze(0)
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            total += test_target[i].size(0)
            correct += (predicted == test_target[i]).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test data: {accuracy:.2f}%")
    return accuracy

# 4. Example Usage (all in one)
if __name__ == "__main__":
    # Create some dummy test data (REPLACE with your actual test data)
    test_data = torch.randn(20, 12)  # Input size MUST match your model's input
    test_target = torch.randint(0, 2, (20,))

    # Path to your original model
    original_model_path = "model.pth"  # Replace with your .pth file

    # Poison the model
    poisoned_model_path = poison_model(original_model_path, poisoning_percentage=0.05)

    if poisoned_model_path:
        # Test the original (clean) model
        original_accuracy = test_model(original_model_path, test_data, test_target)

        # Test the poisoned model
        poisoned_accuracy = test_model(poisoned_model_path, test_data, test_target)

        # Compare the accuracies
        print(f"Difference in accuracy: {original_accuracy - poisoned_accuracy:.2f}%")
