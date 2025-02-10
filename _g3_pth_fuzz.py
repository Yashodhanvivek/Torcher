import torch
import torch.nn as nn
import random
import os

# 1. Define your model class (or import it) - Replace with your actual model
class MyModel(nn.Module):  # Example model (replace with yours)
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def mutate_pth(input_path, output_path):
    try:
        state_dict = torch.load(input_path)

        # *** Mutation logic here ***
        if random.random() < 0.2:
            # Mutate keys
            keys = list(state_dict.keys())
            if keys:
                key_to_change = random.choice(keys)
                new_key = key_to_change + "_mutated"
                state_dict[new_key] = state_dict.pop(key_to_change)

        elif random.random() < 0.2:
            # Mutate values (example: change a tensor's shape)
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    original_shape = list(value.shape)
                    if original_shape:
                        new_shape = list(original_shape)  # Copy
                        dim_to_change = random.randint(0, len(new_shape) - 1)
                        change_factor = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                        new_size = int(max(1, original_shape[dim_to_change] * change_factor))
                        new_shape[dim_to_change] = new_size
                        try:
                            state_dict[key] = torch.randn(new_shape, dtype=value.dtype, device=value.device)
                            print(f"Shape mutated for {key} from {original_shape} to {new_shape}")
                        except Exception as e:
                            print(f"Shape mutation failed for {key}: {e}")

        elif random.random() < 0.2:
            # Inject NaN values (example)
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    nan_indices = torch.rand(value.shape) < 0.01  # 1% chance
                    value[nan_indices] = float('nan')
                    state_dict[key] = value

        elif random.random() < 0.2:
            # Change dtype
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    new_dtype = random.choice([torch.float16, torch.float32, torch.float64])
                    try:
                        state_dict[key] = value.to(new_dtype)
                        print(f"Dtype changed for {key} to {new_dtype}")
                    except Exception as e:
                        print(f"Dtype change failed for {key}: {e}")

        # Add more mutation strategies here ...

        torch.save(state_dict, output_path)
        return True
    except Exception as e:
        print(f"Mutation failed: {e}")
        return False


def fuzz(input_path, num_iterations=1000):
    for i in range(num_iterations):
        output_path = f"mutated_{i}.pth"
        if mutate_pth(input_path, output_path):
            try:
                # 2. Create an instance of your model
                model = MyModel()  # Replace MyModel with your actual model class

                # 3. Load the state dict (strict=False is CRUCIAL)
                state_dict = torch.load(output_path)
                model.load_state_dict(state_dict, strict=False)

                # 4. Now you can call eval()
                model.eval()

                # ... (rest of your testing logic)
                try:
                    dummy_input = torch.randn(1, 12)  # Example input shape (adjust)
                    output = model(dummy_input)

                    # Example output checks (adapt to your model):
                    if output.shape != torch.Size([1, 2]):  # Example output shape
                        print(f"Output shape mismatch: {output.shape}")
                    # Add more checks here (e.g., compare to expected output)

                    print(f"Load and test successful: {output_path}")

                except Exception as test_exception:
                    print(f"Model test failed: {test_exception}")

            except Exception as load_exception:
                print(f"Load failed: {load_exception}")

            finally:
                os.remove(output_path)  # Clean up

fuzz("model.pth")  # Replace with actual path
