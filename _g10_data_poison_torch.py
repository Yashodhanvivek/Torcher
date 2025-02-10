import torch
import torch.nn as nn
import random
import os
import copy

# 1. Define your model class (REPLACE with your ACTUAL model)
class MyModel(nn.Module):  # Example model - REPLACE THIS
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 2. Data poisoning function (unchanged)
def poison_data(data, target, poison_percentage=0.1):
    num_poison = int(len(data) * poison_percentage)
    poison_indices = random.sample(range(len(data)), num_poison)

    poisoned_data = copy.deepcopy(data)
    poisoned_targets = copy.deepcopy(target)

    for i in poison_indices:
        poisoned_targets[i] = 1 - target[i]
        perturbation = torch.randn_like(data[i]) * 0.1
        poisoned_data[i] = data[i] + perturbation

        print(f"Poisoned data point at index: {i}")

    return poisoned_data, poisoned_targets

# 3. Training function (modified to use poisoned data)
def train(model, data, target, poisoned=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    if poisoned:
        data, target = poison_data(data, target)

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 4. Fuzzing function (modified to use original model and compare)
def fuzz(original_model_path, test_data, test_target):
    try:
        original_model = MyModel()
        original_state_dict = torch.load(original_model_path, weights_only=True)
        original_model.load_state_dict(original_state_dict)
        original_model.eval()
        print("Original Model loaded successfully")
    except Exception as e:
        print(f"Error loading original model: {e}")
        return

    poisoned_model = MyModel()
    train_data = torch.randn(100,5)
    train_target = torch.randint(0,2,(100,))
    train(poisoned_model,train_data,train_target,poisoned=True)
    poisoned_model.eval()

    correct_original = 0
    correct_poisoned = 0

    with torch.no_grad():
        for i in range(len(test_data)):
            input_data = test_data[i].unsqueeze(0)

            original_output = original_model(input_data)
            original_prediction = original_output.argmax(dim=1)
            correct_original += (original_prediction == test_target[i]).sum().item()

            poisoned_output = poisoned_model(input_data)
            poisoned_prediction = poisoned_output.argmax(dim=1)
            correct_poisoned += (poisoned_prediction == test_target[i]).sum().item()

    accuracy_original = correct_original / len(test_data)
    accuracy_poisoned = correct_poisoned / len(test_data)

    print(f"Original Model Accuracy: {accuracy_original}")
    print(f"Poisoned Model Accuracy: {accuracy_poisoned}")

    output_diff = torch.abs(original_output - poisoned_output)
    mean_diff = torch.mean(output_diff)

    print(f"Mean output difference: {mean_diff}")

# 5. Example usage (all in one)
if __name__ == "__main__":
    # Create dummy data (for demonstration - REPLACE with your data)
    test_data = torch.randn(20, 5)
    test_target = torch.randint(0, 2, (20,))

    # Create a dummy model and save it (for demonstration - REPLACE with your model)
    dummy_model = MyModel()
    torch.save(dummy_model.state_dict(), "dummy_model.pth")  # Save the dummy model

    # Call the fuzzing function
    fuzz("dummy_model.pth", test_data, test_target)  # Use the saved dummy model
