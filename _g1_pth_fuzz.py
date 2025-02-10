import torch
import random
import os

def mutate_pth(input_path, output_path):
    try:
        state_dict = torch.load(input_path)

        # *** Mutation logic now CORRECTLY placed here ***
        if random.random() < 0.5:
            # Mutate keys
            keys = list(state_dict.keys())  # Now state_dict is defined!
            if keys:
                key_to_change = random.choice(keys)
                new_key = key_to_change + "_mutated"  # Or generate a completely random key
                state_dict[new_key] = state_dict.pop(key_to_change)
        elif random.random() < 0.5:
            # Mutate values (example: change a tensor's shape)
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    new_shape = list(value.shape)  # Make a copy
                    if new_shape:
                        new_shape = random.randint(1, 10)  # Example shape mutation
                        try:
                            state_dict[key] = torch.randn(new_shape, dtype=value.dtype, device=value.device)  # use random data with the same type and device
                        except Exception as e:
                            print(f"Shape mutation failed for {key}: {e}")  # Handle shape mismatch errors
                            #... other ways to mutate the tensor
        #... more mutation strategies...

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
                torch.load(output_path)  # Attempt to load the mutated model
                print(f"Load successful: {output_path}")  # But check for unexpected behavior!
            except Exception as e:
                print(f"Load failed: {e}")  # Potential bug!
                # Log the error, the mutated file, and potentially the PyTorch version
                #...
            finally:
                os.remove(output_path) # Clean up

fuzz("model.pth") # Replace with the actual path

