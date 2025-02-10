import torch
import random
import os

def mutate_pth(input_path, output_path):
    try:
        state_dict = torch.load(input_path)

        # *** Mutation logic here ***
        if random.random() < 0.2:  # Increased probability for mutation
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
                        dim_to_change = random.randint(0, len(new_shape) - 1) # Choose a random dimension to mutate
                        change_factor = random.choice([-1, 1]) * random.uniform(0.5, 2.0) # Random factor between 0.5 and 2.0
                        new_size = int(max(1, original_shape[dim_to_change] * change_factor)) # Ensure size is at least 1
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
                    # Inject NaN values with a small probability
                    nan_indices = torch.rand(value.shape) < 0.01  # 1% chance for each element
                    value[nan_indices] = float('nan')
                    state_dict[key] = value # Update the state dict

        elif random.random() < 0.2:
            # Change dtype
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                  new_dtype = random.choice([torch.float16, torch.float32, torch.float64])
                  try:
                    state_dict[key] = value.to(new_dtype) # Convert to new dtype
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
                model = torch.load(output_path)
                model.eval()

                # Example: Test with some dummy input (adapt to your model)
                try:
                    dummy_input = torch.randn(1, 3, 224, 224)  # Example input shape
                    output = model(dummy_input)

                    # Check output shape and other properties (adapt to your model)
                    if output.shape != torch.Size([1, 1000]):  # Example expected output shape
                        print(f"Output shape mismatch: {output.shape}")
                    # ... more output checks ...

                    print(f"Load and test successful: {output_path}")

                except Exception as test_exception:
                  print(f"Model test failed: {test_exception}")

            except Exception as load_exception:
                print(f"Load failed: {load_exception}")
            finally:
                os.remove(output_path)  # Clean up

fuzz("model.pth")  # Replace with actual path
