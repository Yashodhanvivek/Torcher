import torch
import torch.nn as nn
import copy
import inspect

def extract_model_details(model_path):
    """Extracts detailed information from a PyTorch model."""
    try:
        state_dict = torch.load(model_path, weights_only=True)
    except FileNotFoundError:
        return {"error": f"Model file not found at {model_path}"}
    except Exception as e:
        return {"error": f"Error loading model: {e}"}

    details = {}
    details["architecture"] = []
    for key in state_dict:
        if "weight" in key or "bias" in key or "running_mean" in key or "running_var" in key or "tracked" in key:
            layer_name = ".".join(key.split(".")[:-1])
            layer_type = ""
            if "conv" in layer_name:
                layer_type = "Conv2d"
            elif "linear" in layer_name or "fc" in layer_name:
                layer_type = "Linear"
            elif "batchnorm" in layer_name:
                layer_type = "BatchNorm2d"
            elif "relu" in layer_name:
                layer_type = "ReLU"
            elif "maxpool" in layer_name:
                layer_type = "MaxPool2d"
            elif "dropout" in layer_name:
                layer_type = "Dropout"
            elif "adaptiveavgpool" in layer_name:
                layer_type = "AdaptiveAvgPool2d"
            elif "avgpool" in layer_name:
                layer_type = "AvgPool2d"
            elif "embedding" in layer_name:
                layer_type = "Embedding"
            elif "lstm" in layer_name:
                layer_type = "LSTM"
            else:
                layer_type = "Other"

            shape = state_dict[key].shape
            details["architecture"].append({
                "layer_name": layer_name,
                "layer_type": layer_type,
                "shape": shape,
                "param_type": key.split(".")[-1]
            })

    input_size = None
    output_size = None
    for layer in details["architecture"]:
        if layer["layer_type"] == "Linear" and layer["param_type"] == "weight":
            if input_size is None:
                input_size = layer["shape"][1]
            output_size = layer["shape"][0]
    details["input_size"] = input_size
    details["output_size"] = output_size

    total_params = 0
    trainable_params = 0
    for tensor in state_dict.values():
        total_params += tensor.numel()
        if tensor.requires_grad:
            trainable_params += tensor.numel()

    details["num_parameters"] = total_params
    details["trainable_parameters"] = trainable_params

    activation_functions = {}
    for layer in details["architecture"]:
        if layer["layer_type"] == "ReLU":
            activation_functions[layer["layer_name"]] = "ReLU"
        elif layer["layer_type"] == "Other" and "relu" in layer["layer_name"].lower():
            activation_functions[layer["layer_name"]] = "ReLU"
        elif layer["layer_type"] == "Other" and "sigmoid" in layer["layer_name"].lower():
            activation_functions[layer["layer_name"]] = "Sigmoid"
        elif layer["layer_type"] == "Other" and "tanh" in layer["layer_name"].lower():
            activation_functions[layer["layer_name"]] = "Tanh"

    details["activation_functions"] = activation_functions

    try:
        dummy_input = torch.randn(1, details["input_size"])

        class TempModel(nn.Module):
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

        temp_model = TempModel(details["input_size"])  # Input size is still 12 (example)
        temp_model.load_state_dict(state_dict, strict=False)
        temp_model.eval()

        with torch.no_grad():
            output = temp_model(dummy_input)
            details["forward_pass_output_shape"] = output.shape

    except Exception as e:
        details["forward_pass_error"] = f"Error during forward pass: {e}"

    return details


def weight_flipping_attack(model_path, attack_percentage=0.1, flip_value=1.0):
    try:
        state_dict = torch.load(model_path, weights_only=True)
    except FileNotFoundError:
        return {"error": f"Model file not found at {model_path}"}
    except Exception as e:
        return {"error": f"Error loading model: {e}"}

    attacked_state_dict = copy.deepcopy(state_dict)

    for key, tensor in attacked_state_dict.items():
        if "weight" in key:
            num_elements = tensor.numel()
            num_attacked = int(num_elements * attack_percentage)
            indices = torch.randperm(num_elements)[:num_attacked]
            tensor.view(-1)[indices] = tensor.view(-1)[indices] * -1
            print(f"Flipped {num_attacked} weights in {key}")

    attacked_model_path = "attacked_" + model_path
    torch.save(attacked_state_dict, attacked_model_path)
    return attacked_model_path


def test_model(model_path, input_data, target_data):
    try:
        details = extract_model_details(model_path)
        if "error" in details:
            return {"error": details["error"]}

        class TempModel(nn.Module):
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

        model = TempModel(details["input_size"])
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(input_data)):
                input_ = input_data[i].unsqueeze(0)
                target = target_data[i]
                output = model(input_)
                _, predicted = torch.max(output.data, 1)
                total += 1
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    except Exception as e:
        return {"error": f"Error during testing: {e}"}


if __name__ == "__main__":
    model_path = "model.pth"  # Replace with your model path

    # 1. Inspect and create TempModel (as before) - CRUCIAL
    # You MUST uncomment and run this once:
    # details = extract_model_details(model_path)
    # print(details["architecture"]) # Copy this output
    # Then, create the TempModel class with the correct layers
    # Example (replace with your actual layers):
    # class TempModel(nn.Module):
    #     def __init__(self, input_size):
    #         super().__init__()
    #         self.fc1 = nn.Linear(12, 10)  # Input size MUST be 12 (example)
    #         self.fc2 = nn.Linear(10, 5)
    #         self.fc3 = nn.Linear(5, 2)
    #
    #     def forward(self, x):
    #         x = self.fc1
