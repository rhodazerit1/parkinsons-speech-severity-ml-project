import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# loading the dataset
data = pd.read_csv("../data/parkinsons_updrs.data.csv")

# target: motor_UPDRS
y = data["motor_UPDRS"].values.reshape(-1, 1)

# drop target columns
# drop total_UPDRS so the model is not "cheating"
X = data.drop(columns=["motor_UPDRS", "total_UPDRS"]).values

# standardize X 
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std

# standardize y 
y_mean = y.mean(axis=0)
y_std = y.std(axis=0)
y_std[y_std == 0] = 1
y_scaled = (y - y_mean) / y_std


# train/test split
n = X.shape[0]
indices = np.random.permutation(n)

train_size = int(0.8 * n)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = X[train_idx]
y_train = y_scaled[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]   

# neural network setup
# number of input features
d = X_train.shape[1]   
# number of hidden nodes
q = 8                 
# learning rate
eta = 0.001            
epochs = 300

# weights
W1 = np.random.randn(d, q) * 0.1
b1 = np.zeros((1, q))

W2 = np.random.randn(q, 1) * 0.1
b2 = np.zeros((1, 1))


# activation functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# forward pass
def predict_scaled(X_input):
    Z1 = X_input @ W1 + b1
    H = relu(Z1)
    y_hat = H @ W2 + b2
    return y_hat

# gradient descent training
errors = []
n_train = X_train.shape[0]

for epoch in range(epochs):
    # forward pass
    Z1 = X_train @ W1 + b1
    H = relu(Z1)
    y_hat = H @ W2 + b2

    # loss = mean squared error
    error = y_hat - y_train
    mse = np.mean(error ** 2)
    errors.append(mse)

    # backward pass
    d_y_hat = (2 / n_train) * error
    dW2 = H.T @ d_y_hat
    db2 = np.sum(d_y_hat, axis=0, keepdims=True)

    dH = d_y_hat @ W2.T
    dZ1 = dH * relu_derivative(Z1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # update weights
    W2 -= eta * dW2
    b2 -= eta * db2
    W1 -= eta * dW1
    b1 -= eta * db1

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: MSE = {mse:.4f}")

# test set evaluation
y_pred_scaled = predict_scaled(X_test)

# convert predictions back to original motor_UPDRS scale
y_pred = y_pred_scaled * y_std + y_mean

mse_test = np.mean((y_pred - y_test) ** 2)
rmse_test = np.sqrt(mse_test)
mae_test = np.mean(np.abs(y_pred - y_test))

# simple R^2
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\nTest Results")
print("------------")
print(f"MSE:  {mse_test.item():.4f}")
print(f"RMSE: {rmse_test.item():.4f}")
print(f"MAE:  {mae_test.item():.4f}")
print(f"R^2:  {r2.item():.4f}")

# plotting training loss
plt.plot(range(epochs), errors)
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Manual MLP Training Loss")
plt.show()
