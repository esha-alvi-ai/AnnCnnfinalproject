
#ESHA ALVI     
# WITH ADAM


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load and preprocess dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the ANN class from scratch
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
        # First hidden layer
        x = torch.matmul(x, self.weight1) + self.bias1
        x = torch.relu(x)
        # Second hidden layer
        x = torch.matmul(x, self.weight2) + self.bias2
        x = torch.relu(x)
        # Output layer
        x = torch.matmul(x, self.weight3) + self.bias3
        x = torch.sigmoid(x) 
        return x

# Hyperparameters
input_size = X_train.shape[1]
hidden_size1 = 32
hidden_size2 = 64
output_size = 1  
learning_rate = 0.001
num_epochs = 40
checkpoint_path = "model_checkpoint.pth"


model = SimpleANN(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}!")

train_losses = []

for epoch in range(num_epochs):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    train_losses.append(loss.item())

    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    save_checkpoint(model, optimizer, epoch, checkpoint_path)



# Plot training loss
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Evaluation
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))




with torch.no_grad():
    
    y_pred_train = torch.round(model(X_train))
    y_pred_test = torch.round(model(X_test))

    
    print("Training Metrics:")
    evaluate_metrics(y_train.numpy(), y_pred_train.numpy())

    print("Testing Metrics:")
    evaluate_metrics(y_test.numpy(), y_pred_test.numpy())
def plot_confusion_matrix(cm):

    categories = ['Class 0', 'Class 1']
    bar_width = 0.35
    index = np.arange(2)

    plt.figure(figsize=(6, 4))
    plt.bar(index, cm[0], bar_width, label='Predicted Class 0', color='b')
    plt.bar(index + bar_width, cm[1], bar_width, label='Predicted Class 1', color='r')

    plt.xlabel('Actual Classes')
    plt.ylabel('Counts')
    plt.title('Confusion Matrix')
    plt.xticks(index + bar_width / 2, categories)
    plt.legend()
    plt.show()
    




print("Plotting Testing Confusion Matrix:")
cm_test = confusion_matrix(y_test.numpy(), y_pred_test.numpy())
plot_confusion_matrix(cm_test)
