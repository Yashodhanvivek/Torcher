import torch
import torch.nn as nn
import numpy as np
from scipy.stats import chi2_contingency  # For Chi-squared test

# 1. Define your model class (REPLACE with your ACTUAL model)
class MyModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 2. Prediction analysis function (with Chi-squared test)
def prediction_analysis(model_path, data1, data2):
    try:
        state_dict = torch.load(model_path, weights_only=True)
        input_size = state_dict['fc1.weight'].shape[1]
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

    # Chi-squared test
    observed = []
    for cls in sorted(set(class_distribution1.keys()).union(class_distribution2.keys())): # Handle missing classes
        count1 = class_distribution1.get(cls, 0) # Default to 0 if class is missing
        count2 = class_distribution2.get(cls, 0)
        observed.append([count1, count2])

    observed = np.array(observed)
    if observed.shape == (0,2): # Handle the case where there is no prediction for both data
        print("No prediction for both data")
        return

    chi2, p, dof, expected = chi2_contingency(observed)

    print(f"Chi-squared statistic: {chi2}")
    print(f"P-value: {p}")

    alpha = 0.05
    if p < alpha:
        print("Reject the null hypothesis: Class distributions are significantly different.")
        print("Potential anomaly detected.")
    else:
        print("Fail to reject the null hypothesis: Class distributions are not significantly different.")


# 4. Example usage (all in one)
if __name__ == "__main__":
    data1 = torch.randn(100, 12)  # Example: 100 samples, 12 features (MATCHES fc1 input)
    data2 = torch.randn(100, 12) + torch.randn(100,12) * 0.5 # Example: data with added noise

    model_path = "model.pth"  # Replace with the actual path to your .pth file

    prediction_analysis(model_path, data1, data2)
