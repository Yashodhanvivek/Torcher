import torch
import torch.nn as nn

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = MyModel()

# Access and modify weights using state_dict
state_dict = model.state_dict()
print("Original fc1.weight:", state_dict['fc1.weight'])

# Modify the weights (example: set all to 1.0)
state_dict['fc1.weight'].fill_(1.0)  # In-place modification
print("Modified fc1.weight:", state_dict['fc1.weight'])

model.load_state_dict(state_dict) # load modified state_dict to the model

# Access and modify weights directly (parameters)
print("Original fc2.bias:", model.fc2.bias)
model.fc2.bias.data.fill_(0.5)
print("Modified fc2.bias:", model.fc2.bias)

# Save the modified model
torch.save(model.state_dict(), "modified_model.pth")

# Load the model
loaded_model = MyModel()
loaded_model.load_state_dict(torch.load("modified_model.pth"))
print("loaded model fc1.weight: ", loaded_model.fc1.weight)
print("loaded model fc2.bias: ", loaded_model.fc2.bias)
