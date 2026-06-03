import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------- DATASET ----------------
X, y = make_classification(
    n_samples=1500,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_classes=3,
    class_sep=1.2,
    flip_y=0.08,
    random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ---------------- MODEL ----------------
class ANN_Model(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)
        self.activation = activation_function


    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        return x

# Activation functions
activation_functions = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh()
}

results = {} 
plt.figure(figsize=(8,5))

# ---------------- TRAINING ----------------
for name, activation in activation_functions.items():
    model = ANN_Model(activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    losses = []

    for epoch in range(120):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Evaluate accuracy
    with torch.no_grad():
        predictions = model(X_test)
        _, predicted = torch.max(predictions, 1)
        accuracy = accuracy_score(y_test, predicted)
        results[name] = accuracy

    plt.plot(losses, label=name)

plt.title("Loss Comparison Across Activation Functions")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ---------------- ACCURACY RESULTS ----------------
print("\nAccuracy Comparison")
for k, v in results.items():
    print(f"{k}: {v*100:.2f}%")

# ---------------- FINAL INFERENCE OUTPUTS ----------------
print("\nInference Results (3 Samples Per Activation Function)\n")

for name, activation in activation_functions.items():
    print(f"Activation Function Used: {name}")
    model = ANN_Model(activation)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    # retrain model briefly
    for epoch in range(120):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        sample_inputs = X_test[:3]
        predictions = model(sample_inputs)
        _, predicted_classes = torch.max(predictions, 1)

    for i in range(len(sample_inputs)):
        sample_list = sample_inputs[i].tolist()
        if len(sample_list) > 5:
            display_list = sample_list[:5] + ['...']
        else:
            display_list = sample_list
        print(f"  Input Sample {i+1}: {display_list}")
        print(f"  Predicted Class {i+1}: {predicted_classes[i].item()}")

    print("-" * 40)