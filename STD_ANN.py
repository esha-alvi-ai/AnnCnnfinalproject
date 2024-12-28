import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd



# By CYBIL
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"Dataset: California Housing")
print(f"Features: {X.shape[1]}")
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Target Range: {y.min()} - {y.max()}")

# ANN model
class ManualANNRegression(nn.Module):
    def __init__(self, input_dim):
        super(ManualANNRegression, self).__init__()
        # Layer 1 weights and biases input
        self.weights1 = nn.Parameter(torch.randn(input_dim, 128) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(128))
        
        # Layer 2 hidden layer
        self.weights2 = nn.Parameter(torch.randn(128, 64) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(64))
        
        # Layer 3 hidden layer
        self.weights3 = nn.Parameter(torch.randn(64, 32) * 0.01)
        self.bias3 = nn.Parameter(torch.zeros(32))
        
        # Output layer
        self.weights4 = nn.Parameter(torch.randn(32, 1) * 0.01)
        self.bias4 = nn.Parameter(torch.zeros(1))
        
        # Dropout Layers with dropout 30% rate
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        
       
    
    def forward(self, x):
        # Layer 1: Linear -> ReLU
        x = torch.matmul(x, self.weights1) + self.bias1
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Layer 2: Linear -> ReLU
        x = torch.matmul(x, self.weights2) + self.bias2
        x = torch.relu(x)
        
        # Layer 3: Linear -> ReLU
        x = torch.matmul(x, self.weights3) + self.bias3
        x = torch.relu(x)
        
        # Output Layer: Linear
        x = torch.matmul(x, self.weights4) + self.bias4
        return x

# Instantiate the model
input_dim = X_train.shape[1]
model = ManualANNRegression(input_dim)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  

epochs = 100
batch_size = 1  
patience = 5  # added early stopping patience

train_losses = []
val_losses = []
best_val_loss = float('inf')
no_improvement_epochs = 0

checkpoint_path = "checkpoint.pth"


if os.path.exists(checkpoint_path):
 
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming from epoch {start_epoch+1}")
else:
   
    start_epoch = 0
    best_val_loss = float('inf')
    print("No checkpoint found. Starting training from scratch.")

start_time = time.time()

for epoch in range(start_epoch, epochs):
    model.train()
    epoch_loss = 0
    
   
    for i in range(X_train.size(0)):
       
        batch_X = X_train[i].unsqueeze(0)
        batch_y = y_train[i].unsqueeze(0)
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(X_train))
    
    # Validation loss
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test)
        val_loss = criterion(val_predictions, y_test)
        val_losses.append(val_loss.item())
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_epochs = 0
        # Save checkpoint if validation loss improves
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

training_time = time.time() - start_time

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test).item()
    mae = mean_absolute_error(y_test.numpy(), predictions.numpy())

print(f"Final Metrics For SGD Technique: MSE = {mse:.4f}, MAE = {mae:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

# Plotting
plt.figure(figsize=(10, 5))
epochs_range = range(1, len(train_losses) + 1) 
plt.plot(epochs_range, train_losses, label="Training Loss")
plt.plot(epochs_range[:len(val_losses)], val_losses, label="Validation Loss")  # Adjust to match the number of validation losses
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curves (Stochastic Training)")
plt.legend()
plt.grid()
plt.savefig("learning_curves_stochastic.png")
plt.show()

    

