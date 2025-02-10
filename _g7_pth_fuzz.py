import torch
import torch.nn as nn
import random
import os
import types

# 1. Define your model class (Replace with your actual model)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)
    def forward(self, x):  # The forward method
        # Define how the input x is processed by your model
        x = self.fc1(x)  # Example: Pass x through the first fully connected layer
        x = self.fc2(x) # Example: Pass x through the second fully connected layer
        x = self.fc3(x) # Example: Pass x through the third fully connected layer
        return x  # Return the output    
def mutate_pth(input_path, output_path):
    try:
        state_dict = torch.load(input_path)

        # *** Code Injection (Advanced and Risky - Use with extreme caution) ***
        if random.random() < 0.01:  # Low probability
            for key, value in state_dict.items():
                if isinstance(value, types.FunctionType):
                    try:
                        malicious_code = """
                        def injected_function(x):
                            import os
                            os.system('touch /tmp/pwned')  # Example - VERY DANGEROUS
                            return x
                        """
                        exec(malicious_code, globals())
                        state_dict[key] = injected_function
                        print(f"Injected code into function: {key}")
                    except Exception as e:
                        print(f"Code injection failed: {e}")

        # *** Other Mutations (Less Risky) ***
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                if random.random() < 0.2:  # Shape mutation
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
    original_model = MyModel()
    original_model.load_state_dict(torch.load(input_path))
    original_model.eval()

    for i in range(num_iterations):
        output_path = f"mutated_{i}.pth"
        if mutate_pth(input_path, output_path):
            try:
                model = MyModel()
                state_dict = torch.load(output_path)
                model.load_state_dict(state_dict, strict=False)
                model.eval()

                try:
                    dummy_input = torch.randn(1, 12)
                    output = model(dummy_input)

                    original_output = original_model(dummy_input)
                    diff = torch.abs(output - original_output)
                    mean_diff = torch.mean(diff)

                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print("NaN/Inf values in output!")
                    elif mean_diff > 0.1:
                        print(f"Significant output difference: {mean_diff}")

                    if os.path.exists("/tmp/pwned"):  # Example - VERY DANGEROUS
                        print("Vulnerability found: Code injection successful!")
                        os.remove("/tmp/pwned")
                        break

                    print(f"Load and test successful: {output_path}")

                except Exception as test_exception:
                    print(f"Model test failed: {test_exception}")

            except Exception as load_exception:
                print(f"Load failed: {load_exception}")

            finally:
                os.remove(output_path)

fuzz("model.pth") # Replace with the actual path
