import torch
import torch.nn as nn
import random
import os
import types

# 1. Define your model class (REPLACE with your ACTUAL model definition)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example:  REPLACE these with your actual layers and sizes
        self.fc1 = nn.Linear(5, 10)  # Input 5, Output 10 (Example)
        self.fc2 = nn.Linear(10, 2)  # Input 10, Output 2 (Example)
        # Add other layers (fc3, etc.) if they were in the original model
        # Example: self.fc3 = nn.Linear(2, 3)  # If original model had fc3

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # Use other layers in forward if needed
        # x = self.fc3(x)
        return x

def mutate_pth(input_path, output_path):
    try:
        state_dict = torch.load(input_path, weights_only=True)

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Your mutation logic here (shape, dtype, value changes, key deletion)
                if random.random() < 0.2:  # Shape mutation example
                    original_shape = list(value.shape)
                    new_shape = list(original_shape)
                    dim_to_change = random.randint(0, len(new_shape) - 1)
                    change_factor = random.choice([-0.5, 0.5, 1, 2])
                    new_size = int(max(1, original_shape[dim_to_change] * change_factor))
                    new_shape[dim_to_change] = new_size
                    try:
                        state_dict[key] = torch.randn(new_shape, dtype=value.dtype, device=value.device)
                        print(f"Shape mutated for {key} from {original_shape} to {new_shape}")
                    except Exception as e:
                        print(f"Shape mutation failed for {key}: {e}")

                elif random.random() < 0.2:  # NaN/Inf injection
                    nan_indices = torch.rand(value.shape) < 0.05
                    value[nan_indices] = float('nan')
                    inf_indices = torch.rand(value.shape) < 0.05
                    value[inf_indices] = float('inf')
                    state_dict[key] = value

                elif random.random() < 0.2:  # Data type change
                    new_dtype = random.choice([torch.float16, torch.float32, torch.float64])
                    try:
                        state_dict[key] = value.to(new_dtype)
                        print(f"Dtype changed for {key} to {new_dtype}")
                    except Exception as e:
                        print(f"Dtype change failed for {key}: {e}")

                elif random.random() < 0.1:  # Value replacement
                    try:
                        state_dict[key] = torch.randn_like(value)
                        print(f"Values replaced for {key}")
                    except Exception as e:
                        print(f"Value replacement failed for {key}: {e}")

                elif random.random() < 0.1:  # Delete a key
                    if random.choice(list(state_dict.keys())) == key:
                        del state_dict[key]
                        print(f"Key {key} deleted")

        torch.save(state_dict, output_path)
        return True
    except Exception as e:
        print(f"Mutation failed: {e}")
        return False


def fuzz(input_path, num_iterations=1000):
    try:  # Try to load the original model *once*
        original_model = MyModel()  # Create model instance
        original_state_dict = torch.load(input_path, weights_only=True)
        original_model.load_state_dict(original_state_dict)  # Load state dict
        original_model.eval()
    except RuntimeError as e:
        print(f"Error loading original model: {e}")
        return  # Exit if original model loading fails

    for i in range(num_iterations):
        output_path = f"mutated_{i}.pth"
        if mutate_pth(input_path, output_path):
            try:
                model = MyModel()  # New model instance *inside* the loop
                state_dict = torch.load(output_path, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                model.eval()

                try:
                    dummy_input = torch.randn(1, 5)  # Input size MUST match model
                    output = model(dummy_input)

                    original_output = original_model(dummy_input)
                    diff = torch.abs(output - original_output)
                    mean_diff = torch.mean(diff)

                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print("NaN/Inf values in output!")
                    elif mean_diff > 0.1:
                        print(f"Significant output difference: {mean_diff}")

                    print(f"Load and test successful: {output_path}")

                except Exception as test_exception:
                    print(f"Model test failed: {test_exception}")
                    if isinstance(test_exception, TypeError):
                        print("Potential type error in model")
                    if isinstance(test_exception, RuntimeError) and "size mismatch" in str(test_exception):
                        print("Size mismatch during model forward pass")  # Check during forward pass

            except Exception as load_exception:
                print(f"Load failed: {load_exception}")
                if isinstance(load_exception, RuntimeError) and "size mismatch" in str(load_exception):
                    print("Size mismatch during loading (expected during fuzzing)")
                elif isinstance(load_exception, KeyError):
                    print("KeyError during loading (possible missing key)")

            finally:
                os.remove(output_path)  # Clean up

fuzz("model.pth")  # Replace with actual path
