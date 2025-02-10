import torch
import torch.nn as nn

def extract_model_details(model_path):
    """Extracts detailed information from a PyTorch model."""

    try:
        state_dict = torch.load(model_path, weights_only=True)
    except FileNotFoundError:
        return {"error": f"Model file not found at {model_path}"}
    except Exception as e:
        return {"error": f"Error loading model: {e}"}

    details = {}

    # 1. Architecture Details (More Comprehensive)
    details["architecture"] = []
    for key in state_dict:
        if "weight" in key or "bias" in key:  # Consider both weights and biases
            layer_name = ".".join(key.split(".")[:-1]) # Extract Layer name
            layer_type = ""
            if "conv" in layer_name:
                layer_type = "Conv2d" # Assuming 2D Convolution
            elif "linear" in layer_name or "fc" in layer_name:
                layer_type = "Linear"
            elif "batchnorm" in layer_name:
                layer_type = "BatchNorm2d" # Assuming 2D Batchnorm
            elif "relu" in layer_name:
                layer_type = "ReLU"
            elif "maxpool" in layer_name:
                layer_type = "MaxPool2d"
            else:
                layer_type = "Other"

            shape = state_dict[key].shape
            details["architecture"].append({
                "layer_name": layer_name,
                "layer_type": layer_type,
                "shape": shape,
            })

    # 2. Input Size (More Robust)
    input_size = None
    for layer in details["architecture"]:
        if layer["layer_type"] == "Linear" and "weight" in layer["layer_name"]:  # Check for Linear layers
            input_size = layer["shape"][1] # Get input size from weights of Linear Layer
            break # Get from first linear layer
    details["input_size"] = input_size

    # 3. Parameter Count (Trainable and Non-Trainable)
    total_params = 0
    trainable_params = 0
    for tensor in state_dict.values():
        total_params += tensor.numel()
        if tensor.requires_grad:
            trainable_params += tensor.numel()

    details["num_parameters"] = total_params
    details["trainable_parameters"] = trainable_params

    # 4. Activation Functions (Inferred, limited)
    activation_functions = {}
    for layer in details["architecture"]:
        if layer["layer_type"] == "ReLU":
            activation_functions[layer["layer_name"]] = "ReLU"
        # Add more activation functions as needed (e.g., sigmoid, tanh)

    details["activation_functions"] = activation_functions

    # 5. Output Size (Inferred, limited, from the last linear layer)
    output_size = None
    for layer in reversed(details["architecture"]):  # Check from the last layer back
        if layer["layer_type"] == "Linear" and "weight" in layer["layer_name"]:
            output_size = layer["shape"][0] # Get output size from weights of Linear Layer
            break
    details["output_size"] = output_size

    return details



# Example usage:
model_path = "model.pth"  # Replace with your model path
details = extract_model_details(model_path)

if "error" in details:
    print(details["error"])
else:
    print("Model Details:")
    for key, value in details.items():
        print(f"{key}: {value}")
