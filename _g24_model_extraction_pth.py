import torch

def extract_model_details(model_path):
    """
    Extracts important details from a PyTorch model saved as a .pth file.

    Args:
        model_path (str): The path to the .pth file.

    Returns:
        dict: A dictionary containing model details, or None if an error occurs.
              The dictionary may contain the following keys (not all are guaranteed to be present):
                  - input_size: The input size of the first linear layer (if found).
                  - layer_names: A list of layer names.
                  - layer_shapes: A dictionary mapping layer names to their weight shapes.
                  - num_parameters: The total number of parameters in the model.
                  - trainable_parameters: The number of trainable parameters.
    """
    try:
        state_dict = torch.load(model_path, weights_only=True)  # Load state_dict
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model_details = {}

    # 1. Input Size (try to infer from the first linear layer)
    for key in state_dict:
        if "fc1.weight" in key or "linear1.weight" in key or "conv1.weight" in key: # Common names
            try:
                input_size = state_dict[key].shape[1]
                model_details["input_size"] = input_size
                break  # Stop after finding the first one
            except:
                pass

    # 2. Layer Names and Shapes
    layer_names = []
    layer_shapes = {}
    for key, tensor in state_dict.items():
        name = key.split('.')[0]  # Extract layer name (e.g., fc1, conv2)
        if name not in layer_names: # Avoid duplicates
            layer_names.append(name)
        if 'weight' in key: # Only save weight shapes
            layer_shapes[name] = tensor.shape

    model_details["layer_names"] = layer_names
    model_details["layer_shapes"] = layer_shapes

    # 3. Parameter Count
    total_params = 0
    trainable_params = 0
    for tensor in state_dict.values():
        total_params += tensor.numel()
        if tensor.requires_grad:  # Check if gradient is required
            trainable_params += tensor.numel()

    model_details["num_parameters"] = total_params
    model_details["trainable_parameters"] = trainable_params

    return model_details


# Example usage:
model_path = "model.pth"  # Replace with your model path
details = extract_model_details(model_path)

if details:
    print("Model Details:")
    for key, value in details.items():
        print(f"{key}: {value}")
else:
    print("Failed to extract model details.")
