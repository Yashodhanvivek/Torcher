import torch
import torch.nn as nn
import copy

# ... (extract_model_details function - same as before)

def weight_flipping_attack(model_path, attack_percentage=0.1, flip_value=1.0):
    """Performs a weight-flipping attack on a PyTorch model."""

    try:
        state_dict = torch.load(model_path, weights_only=True)
    except FileNotFoundError:
        return {"error": f"Model file not found at {model_path}"}
    except Exception as e:
        return {"error": f"Error loading model: {e}"}

    attacked_state_dict = copy.deepcopy(state_dict)  # Crucial: Deep copy!

    for key, tensor in attacked_state_dict.items():
        if "weight" in key:  # Only attack weight tensors
            num_elements = tensor.numel()
            num_attacked = int(num_elements * attack_percentage)
            indices = torch.randperm(num_elements)[:num_attacked]  # Random indices

            # Flip the weights (you can modify the flip value)
            tensor.view(-1)[indices] = tensor.view(-1)[indices] * -1 # Or any other manipulation

            print(f"Flipped {num_attacked} weights in {key}")

    attacked_model_path = "attacked_" + model_path  # Save the attacked model
    torch.save(attacked_state_dict, attacked_model_path)
    return attacked_model_path


def test_model(model_path, input_data, target_data):
    """Tests a PyTorch model and returns the accuracy."""

    try:
        details = extract_model_details(model_path)
        if "error" in details:
            return {"error": details["error"]}

        # Create a dummy model instance (HARDCODED forward pass - must match architecture)
        class TempModel(nn.Module): # Same TempModel as earlier
            def __init__(self, input_size):
                super().__init__()
                self.fc1 = nn.Linear(12, 10)  # Input size MUST be 12 (example)
                self.fc2 = nn.Linear(10, 5)
                self.fc3 = nn.Linear(5, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                return x

        model = TempModel(details["input_size"])  # Input size is still 12 (example)
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(input_data)):
                input_ = input_data[i].unsqueeze(0)  # Add batch dimension
                target = target_data[i]
                output = model(input_)
                _, predicted = torch.max(output.data, 1)
                total += 1 # target.size(0) # Assuming batch size of 1
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    except Exception as e:
        return {"error": f"Error during testing: {e}"}




# Example usage:
if __name__ == "__main__":
    model_path = "model.pth"  # Replace with your model path

    # 1. Inspect and create TempModel (as before) - CRUCIAL

    # 2. Perform the weight-flipping attack
    attacked_model_path = weight_flipping_attack(model_path, attack_percentage=0.05) # 5% attack

    if "error" in attacked_model_path:
        print(attacked_model_path["error"])
    else:
        print(f"Attacked model saved to: {attacked_model_path}")

        # 3. Create dummy input data for testing
        input_size = 12 # From the architecture
        num_samples = 100
        input_data = torch.randn(num_samples, input_size)
        target_data = torch.randint(0, 2, (num_samples,)) # Assuming 2 classes

        # 4. Test the original model
        original_accuracy = test_model(model_path, input_data, target_data)
        if "error" in original_accuracy:
            print(original_accuracy["error"])
        else:
            print(f"Original model accuracy: {original_accuracy:.2f}%")

        # 5. Test the attacked model
        attacked_accuracy = test_model(attacked_model_path, input_data, target_data)
        if "error" in attacked_accuracy:
            print(attacked_accuracy["error"])
        else:
            print(f"Attacked model accuracy: {attacked_accuracy:.2f}%")

        # 6. Compare the accuracies
        if isinstance(original_accuracy, float) and isinstance(attacked_accuracy, float):
            print(f"Accuracy drop: {original_accuracy - attacked_accuracy:.2f}%")
