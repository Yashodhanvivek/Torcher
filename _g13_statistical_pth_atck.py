import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ks_2samp  # For Kolmogorov-Smirnov test

# 1. Define a simple model (REPLACE with your actual model)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 2. Function to generate model outputs
def get_model_outputs(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(len(data)):
            input_data = data[i].unsqueeze(0)  # Add batch dimension
            output = model(input_data)
            outputs.append(output.numpy())  # Store as NumPy array
    return np.concatenate(outputs)  # Concatenate all outputs

# 3. Statistical attack function (Kolmogorov-Smirnov test example)
def statistical_attack(model_path, data1, data2):
    try:
        model = MyModel()
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    outputs1 = get_model_outputs(model, data1)
    outputs2 = get_model_outputs(model, data2)

    # Example: KS test on the first output dimension
    ks_statistic, p_value = ks_2samp(outputs1[:, 0], outputs2[:, 0])

    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("Reject the null hypothesis: Distributions are significantly different.")
        print("Potential anomaly detected.")
    else:
        print("Fail to reject the null hypothesis: Distributions are not significantly different.")

# 4. Example usage
if __name__ == "__main__":
    # Create dummy data (REPLACE with your actual data)
    data1 = torch.randn(100, 5)  # "Normal" data
    data2 = torch.randn(100, 5) + torch.randn(100,5) * 0.5  # "Potentially anomalous" data (added noise)
    test_data = torch.randn(20, 5)
    test_target = torch.randint(0, 2, (20,))

    # Create a dummy model and save it (REPLACE with your model)
    dummy_model = MyModel()  #"model.pth"
    torch.save(dummy_model.state_dict(), "stat_dummy_model.pth")

    # Perform the statistical attack
    statistical_attack("stat_dummy_model.pth", data1, data2) # Compare data1 with data2

    # You can also compare the test data distribution with a known normal distribution
    # statistical_attack("dummy_model.pth", test_data, torch.randn(20,5))
