import torch
import torch.nn as nn
import numpy as np

# 1. Define your model class (REPLACE with your ACTUAL model)
class MyModel(nn.Module):
    def __init__(self, input_size):  # Add input size as parameter
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)  # Input size from inspection
        self.fc2 = nn.Linear(10, 5)  # Output size from inspection
        self.fc3 = nn.Linear(5, 2)  # Output size from inspection

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 2. Function to generate model outputs (unchanged)
def get_model_outputs(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(len(data)):
            input_data = data[i].unsqueeze(0)
            output = model(input_data)
            outputs.append(output.numpy())
    return np.concatenate(outputs)

# 3. Prediction analysis function
def prediction_analysis(model_path, data1, data2):
    try:
        state_dict = torch.load(model_path, weights_only=True)
        input_size = state_dict['fc1.weight'].shape[1]  # Get input size
        model = MyModel(input_size)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"Model loaded successfully from: {model_path}")

    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    predictions1 = []
    predictions2 = []

    with torch.no_grad():
        for i in range(len(data1)):
            input_data = data1[i].unsqueeze(0)
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            predictions1.append(predicted.item())

        for i in range(len(data2)):
            input_data = data2[i].unsqueeze(0)
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            predictions2.append(predicted.item())

    unique1, counts1 = np.unique(predictions1, return_counts=True)
    class_distribution1 = dict(zip(unique1, counts1))

    unique2, counts2 = np.unique(predictions2, return_counts=True)
    class_distribution2 = dict(zip(unique2, counts2))

    print("Class Distribution for data1:", class_distribution1)
    print("Class Distribution for data2:", class_distribution2)

    if 0 in class_distribution1 and 0 in class_distribution2:
        change_in_class0 = class_distribution2[0] - class_distribution1[0]
        print(f"Change in count of class 0: {change_in_class0}")


# 4. Example usage (all in one)
if __name__ == "__main__":
    # 1. Load your data (REPLACE with your actual data loading)
    # Example: Creating dummy data - REPLACE with your data!
    data1 = torch.randn(100, 12)  # Example: 100 samples, 12 features (MATCHES fc1 input)
    data2 = torch.randn(100, 12) + torch.randn(100,12) * 0.5 # Example: data with added noise

    # 2. Provide the path to your .pth model
    model_path = "model.pth"  # Replace with the actual path to your .pth file

    # 3. Perform the analysis
    prediction_analysis(model_path, data1, data2)
