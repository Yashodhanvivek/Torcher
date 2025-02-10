import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ks_2samp

# 1. Define YOUR model class (THIS IS THE MOST IMPORTANT PART)
# REPLACE THIS EXAMPLE WITH YOUR ACTUAL MODEL ARCHITECTURE
class MyModel(nn.Module):  # Example - REPLACE with your actual model
    def __init__(self):
        super().__init__()
        # Example - REPLACE with your actual layers and sizes
        self.fc1 = nn.Linear(12, 10)  # Example: Input 12, Output 10
        self.fc2 = nn.Linear(10, 2)   # Example: Input 10, Output 2
        self.fc3 = nn.Linear(2, 3)    # Example: Input 2, Output 3

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

# 3. Statistical attack function (Kolmogorov-Smirnov test example)
def statistical_attack(model_path, data1, data2):
    try:
        model = MyModel()  # Create an instance of your model
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    outputs1 = get_model_outputs(model, data1)
    outputs2 = get_model_outputs(model, data2)

    ks_statistic, p_value = ks_2samp(outputs1[:, 0], outputs2[:, 0])

    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: Distributions are significantly different.")
        print("Potential anomaly detected.")
    else:
        print("Fail to reject the null hypothesis: Distributions are not significantly different.")

# 4. Example usage (using your existing .pth model)
if __name__ == "__main__":
    # 1. Load your data (REPLACE with your actual data loading)
    # data1: Your "normal" or baseline data
    # data2: The data you want to compare (potentially anomalous)

    # Example: Creating dummy data - REPLACE with your data!
    data1 = torch.randn(100, 12)  # Example: 100 samples, 12 features (MATCHES fc1 input)
    data2 = torch.randn(100, 12) + torch.randn(100,12) * 0.5 # Example: data with added noise

    # 2. Provide the path to your .pth model
    model_path = "model.pth"  # Replace with the actual path to your .pth file

    # 3. Perform the statistical attack
    statistical_attack(model_path, data1, data2)
