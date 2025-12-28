# %% [markdown]
"""
# Zero to Mastery in PyTorch

## Ch 1: PyTorch Workflow Fundamentals

[Link](https://www.learnpytorch.io/01_pytorch_workflow)

### Pytorch Workflow
    * Get dat aready (turn into tensors).
    * Build or pick a pretrained model.
    * Fit the model to the data and make a prediction.
    * Evaluate the model.
    * Improve through experimentation.
    * Save and reload your trained model.
"""

# %%
import torch
from torch import nn
import matplotlib.pyplot as plt

# %% [markdown]
"""
Machine learning is a game of two parts:
    * Turn your data (whatever it may be) into numbers (a representation).
    * Pick or build a model to learn the representation as best as possible.
"""

# %%
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start=start, end=end, step=step)
X = X.unsqueeze(dim=1)
y = weight * X + bias  # X is features, y is labels

X[:10], y[:10]

# %%
len(X), len(y), 1

# %% [markdown]
"""
### Split data into training and test sets

* Training set - model learns from this data (60-80% of data).
* Validation set - model gets tuned on this data (10-20%).
* Test set - model gets evaluated on this data (10-20%).
"""

# %%
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)


# %%
def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


# %%
plot_predictions()


# %%
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float, requires_grad=True)
        )
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# %% [markdown]
"""
## Pytorch model building essentials

Pytorch has 4 essentials modules for neural networks:
    * torch.nn - contians building blocks for computational graphs.
        * torch.nn.Paramater - Stores tensors that can be used with
        nn.Module. If requires_grade=True, gradients are calculated
        automatically, this is often referred to as 'autograd'.
        * torch.nn.Module - Base class for all neural network modules.
        If you're building a neural network, subclass torch.nn.Module.
        Requires forward method to be implemented.
    * torch.optim - Contains various optimization algos, these tell
    nn.Parameter how best to improve gradient descent and in turn reduce
    the loss.
    * torch.utils.data.Dataset
    * torch.utils.data.DataLoader
"""

# %%
torch.manual_seed(seed=42)
model_0 = LinearRegressionModel()
list(model_0.parameters())

# %%
# list named parameters
model_0.state_dict()
