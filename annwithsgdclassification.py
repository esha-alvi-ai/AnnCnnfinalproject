import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load and preprocess dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the ANN class
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleANN, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(input_size, hidden_size1, requires_grad=True))
        self.bias1 = nn.Parameter(torch.zeros(hidden_size1, requires_grad=True))
        self.weight2 = nn.Parameter(torch.randn(hidden_size1, hidden_size2, requires_grad=True))
        self.bias2 = nn.Parameter(torch.zeros(hidden_size2, requires_grad=True))
        self.weight3 = nn.Parameter(torch.randn(hidden_size2, output_size, requires_grad=True))
        self.bias3 = nn.Parameter(torch.zeros(output_size, requires_grad=True))

    def forward(self, x):
        x = torch.matmul(x, self.weight1) + self.bias1
        x = torch.relu(x)
        x = torch.matmul(x, self.weight2) + self.bias2
        x = torch.relu(x)
        x = torch.matmul(x, self.weight3) + self.bias3
        return x  # No sigmoid here as BCEWithLogitsLoss handles it

# Hyperparameters
input_size = X_train.shape[1]
hidden_size1 = 32
hidden_size2 = 64
output_size = 1 
learning_rate = 0.01
num_epochs = 40


# Initialize model, loss function, and optimizer
model = SimpleANN(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.BCEWithLogitsLoss(model.parameter,lr=lr)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    optimizer.zero_grad()
    loss.backward()  # Backpropagation
    optimizer.step()
    train_losses.append(loss.item())
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot training loss
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    return cm

# Evaluate on training and testing data
with torch.no_grad():
    model.eval()
    y_pred_train = torch.round(torch.sigmoid(model(X_train))) 
    y_pred_test = torch.round(torch.sigmoid(model(X_test)))

    print("Training Metrics:")
    cm_train = evaluate_metrics(y_train.numpy(), y_pred_train.numpy())

    print("Testing Metrics:")
    cm_test = evaluate_metrics(y_test.numpy(), y_pred_test.numpy())

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


plot_confusion_matrix(cm_test, title="Test Confusion Matrix")
