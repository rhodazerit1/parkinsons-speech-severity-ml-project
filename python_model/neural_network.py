import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Load dataset
data = pd.read_csv("parkinsons_updrs.data.csv")


# Target variable
y = data["motor_UPDRS"].values.reshape(-1, 1)


# Drop columns we do not want
# motor_UPDRS = target
# total_UPDRS = leakage
# subject# and index = identifiers, not useful predictive features
drop_cols = ["motor_UPDRS", "total_UPDRS", "subject#", "index"]
drop_cols = [col for col in drop_cols if col in data.columns]
X = data.drop(columns=drop_cols).values


# Train/test split
n = X.shape[0]
indices = np.random.permutation(n)

train_size = int(0.8 * n)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train_raw = X[train_idx]
X_test_raw = X[test_idx]

y_train_raw = y[train_idx]
y_test = y[test_idx]


# Standardize X using train only
X_mean = X_train_raw.mean(axis=0)
X_std = X_train_raw.std(axis=0)
X_std[X_std == 0] = 1

X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean) / X_std


# Standardize y using train only
y_mean = y_train_raw.mean(axis=0)
y_std = y_train_raw.std(axis=0)
y_std[y_std == 0] = 1

y_train = (y_train_raw - y_mean) / y_std


# Neural network setup
d = X_train.shape[1]
h1 = 64
h2 = 32
eta = 0.01
epochs = 2000
batch_size = 64

# He initialization
W1 = np.random.randn(d, h1) * np.sqrt(2 / d)
b1 = np.zeros((1, h1))

W2 = np.random.randn(h1, h2) * np.sqrt(2 / h1)
b2 = np.zeros((1, h2))

W3 = np.random.randn(h2, 1) * np.sqrt(2 / h2)
b3 = np.zeros((1, 1))


# Activation functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

# Forward pass
def predict_scaled(X_input):
    Z1 = X_input @ W1 + b1
    H1 = relu(Z1)

    Z2 = H1 @ W2 + b2
    H2 = tanh(Z2)

    y_hat = H2 @ W3 + b3
    return y_hat

# Training loop (mini-batch)
errors = []
n_train = X_train.shape[0]

for epoch in range(epochs):
    perm = np.random.permutation(n_train)
    X_train_epoch = X_train[perm]
    y_train_epoch = y_train[perm]

    for i in range(0, n_train, batch_size):
        X_batch = X_train_epoch[i:i + batch_size]
        y_batch = y_train_epoch[i:i + batch_size]

        # Forward
        Z1 = X_batch @ W1 + b1
        H1 = relu(Z1)

        Z2 = H1 @ W2 + b2
        H2 = tanh(Z2)

        y_hat = H2 @ W3 + b3

        # Loss gradient
        error = y_hat - y_batch
        d_y = (2 / len(X_batch)) * error

        # Backprop
        dW3 = H2.T @ d_y
        db3 = np.sum(d_y, axis=0, keepdims=True)

        dH2 = d_y @ W3.T
        dZ2 = dH2 * tanh_derivative(Z2)
        dW2 = H1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dH1 = dZ2 @ W2.T
        dZ1 = dH1 * relu_derivative(Z1)
        dW1 = X_batch.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update
        W3 -= eta * dW3
        b3 -= eta * db3
        W2 -= eta * dW2
        b2 -= eta * db2
        W1 -= eta * dW1
        b1 -= eta * db1

    # Learning rate decay
    eta *= 0.995

    # Track full training loss
    full_pred = predict_scaled(X_train)
    mse = np.mean((full_pred - y_train) ** 2)
    errors.append(mse)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: MSE = {mse:.4f}")


# Test set evaluation
y_pred_scaled = predict_scaled(X_test)
y_pred = y_pred_scaled * y_std + y_mean

mse_test = np.mean((y_pred - y_test) ** 2)
rmse_test = np.sqrt(mse_test)
mae_test = np.mean(np.abs(y_pred - y_test))

ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\nTest Results")
print("------------")
print(f"MSE:  {mse_test.item():.4f}")
print(f"RMSE: {rmse_test.item():.4f}")
print(f"MAE:  {mae_test.item():.4f}")
print(f"R^2:  {r2.item():.4f}")


# Training loss plot
plt.figure(figsize=(6, 4))
plt.plot(range(epochs), errors)
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Manual MLP Training Loss")
plt.tight_layout()
plt.show()


# Predicted vs Actual plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.25, s=10, color="blue")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="red",
    linewidth=2
)
plt.title("MLP: Predicted vs Actual")
plt.xlabel("Actual motor_UPDRS")
plt.ylabel("Predicted motor_UPDRS")
plt.tight_layout()
plt.show()

# dimension reduction plot
from sklearn.manifold import TSNE

# Use standardized features
X_scaled_full = (X - X.mean(axis=0)) / X.std(axis=0)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled_full)

plt.figure(figsize=(7, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c=data["motor_UPDRS"],
            cmap="viridis", alpha=0.6, s=10)

plt.colorbar(label="motor_UPDRS")
plt.title("t-SNE Representation of Voice Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.show()
