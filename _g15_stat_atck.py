import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ks_2samp

# 1. Inspect the saved model (CRITICAL):
try:
    state_dict = torch.load("model.pth", weights_only=True)  # Replace "model.pth" with your path
    print("Model loaded successfully.  Inspecting architecture:")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()  # Stop execution if loading fails

# 2. Define a *minimal* model (for testing ONLY - REPLACE with your model)
class MyModel(nn.Module):
    def __init__(self, input_size):  # Add input size as parameter
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)  # Example: Input size from inspection
        self.fc2 = nn.Linear(10,5)  # Example: Output 2
        self.fc3 = nn.Linear(5,2)
	#self.fc3 = nn.Linear(5,2)
	
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 3. Attempt to create and load the model (using inspected shapes):
try:
    # Get input size from fc1.weight shape (assuming it exists)
    input_size = state_dict['fc1.weight'].shape[1]  # Get the second element of the shape
    model = MyModel(input_size) # Create an instance of your model
    model.load_state_dict(state_dict, strict=False)  # strict=False is KEY here
    print("Model creation and loading successful (strict=False).")

    #Verify input size
    dummy_input = torch.randn(1,input_size)
    output = model(dummy_input)
    print("Dummy input test successful")


except Exception as e:
    print(f"Error during model creation or loading: {e}")

# 4. If the above works, THEN try your statistical test (simplified):
if 'model' in locals(): # Check if the model was created
    data1 = torch.randn(100, input_size)  # Example data - REPLACE with your data
    data2 = torch.randn(100, input_size) + torch.randn(100,input_size) * 0.5 # Example data with added noise
    # ... (rest of your statistical_attack function - KS test, etc.)
    def get_model_outputs(model, data):
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(len(data)):
                input_data = data[i].unsqueeze(0)
                output = model(input_data)
                outputs.append(output.numpy())
        return np.concatenate(outputs)

    def statistical_attack(model, data1, data2): # Changed parameter to model
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

    statistical_attack(model, data1, data2) # call with the model
