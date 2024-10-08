import torch
from torch import nn
import matplotlib.pyplot as plt

# Create initial parameters
weight = 0.3
bias = 0.9

# Create a list of features and labels
start = 0
end = 1
step = 0.005

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

# Break data up into train and test groupings
train_split = int(len(X) * 0.8)

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Create scatter plots for train, test, and predictions data
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    
    plt.figure("Linear Regression NN", figsize=(10,7))
    plt.title("Linear Regression Model Results")
    plt.xlabel("Features")
    plt.ylabel("Labels")

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

# This will show scatter plots without predictions data
    
# plot_predictions()
# plt.show()
    
# Create the Linear Regression class to run the nn on
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
# Set seed
torch.manual_seed(400)

# Create initial model
model_0 = LinearRegressionModel()

# Create loss function - Mean Average Error
loss_fn = nn.L1Loss()

# Optimizes learning by adjusting learning rate and momentum
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.005, momentum=0.9)

# Set seed
torch.manual_seed(400)

# Number of learning trials
epochs = 300

# Create empty lists to print later
epoch_count = []
train_loss_values = []
test_loss_values = []

# For each trial
for epoch in range(epochs):
    model_0.train()

    # Assigns predictions to variable
    y_pred = model_0(X_train)  

    # Compute MAE from prediction and expected
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad() # Clears gradients
    loss.backward()       # Backward propagation
    optimizer.step()      # Steps through
    model_0.eval()        # Enter evaluation mode

    with torch.inference_mode():
        # Assigns predictions to variable - for test
        test_pred = model_0(X_test)

        # Compute MAE from prediction and expected - for test
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        if epoch % 20 == 0:
            # Add values to previously defined lists
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

model_0.eval()

with torch.inference_mode():
    y_preds = model_0(X_test)

# Create display where prediction values are now visible
plot_predictions(predictions=y_preds)
plt.show()


