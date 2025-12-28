# %% [markdown]
"""
# Zero to Mastery in PyTorch

## Ch 0: PyTorch Fundamentals

[Link](https://www.learnpytorch.io/00_pytorch_fundamentals/#introduction-to-tensors)
"""

# %%
import torch

torch.__version__

# %% [markdown]
"""
## Scalar

Scalar is a single number and a zero dimension tensor.
"""

# %%
scalar = torch.tensor(data=7)
scalar, scalar.ndim  # check dimensions

# %%
scalar = torch.tensor(data=-1)
scalar.item()

# %% [markdown]
"""
## Vectors

Vector is a single dimension tensor.

You can tell number of dimensions in an tensor by number of square
brackets on the outside.
"""

# %%
vector = torch.tensor(data=[7, 7])
vector, vector.ndim, vector.shape
# print(vector.item()) # will error

# %% [markdown]
"""
## Matrix

Matrix is a two dimension tensor.
"""

# %%
matrix = torch.tensor(data=[[7, 8], [9, 10]])
matrix, matrix.ndim, matrix.shape

# %% [markdown]
"""
## Tensor

Tensor is an n-dimension array.

Matrix and Tensor are denoted with uppercase letters, while scalar and
vector are denoted with lowercase letters.
"""

# %%
_3d_tensor = torch.tensor(
    data=[
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]],
    ]
)
_3d_tensor, _3d_tensor.ndim, _3d_tensor.shape

# %% [markdown]
"""
Machine learning models often start out with large random tensor of
number and adjust these random numbers as it works through data to better
represent it.
"""

# %%
# create random tensor of dimension 2 (default dtype is float32)
random_tensor = torch.rand(size=(3, 4))
random_tensor

# %%
# create random tensor of dimension 3 with float16
random_tensor2 = torch.rand(size=(3, 4, 4), dtype=torch.float16)
random_tensor2.shape, random_tensor2.dtype

# %% [markdown]
"""
## Upper/lower triangular matrix
"""

# %%
# return upper triangular part of random tensor
torch.tril(input=random_tensor)

# %%
# return upper triangular part of random tensor
torch.tril(input=random_tensor)

# %% [markdown]
"""
## Zeros and Ones
"""

# %%
# create a tensor of all zeros
zeros = torch.zeros(size=(2, 3))
zeros, zeros.dtype

# %%
# create a tensor of all zeros
ones = torch.ones(size=(2, 3))
ones, ones.dtype

# %% [markdown]
"""
### Creating tensor from range
"""

# %%
zero_to_ten = torch.arange(start=0, end=20, step=2)
zero_to_ten

# %% [markdown]
"""
### Creating a zero/one tensor with the same shape as another tensor
"""

# %%
m = torch.arange(start=0, end=10, step=1)
torch.zeros_like(input=m)  # creates zeros tensor with the same shape as m

# %%
m = torch.arange(start=0, end=10, step=1)
torch.ones_like(input=m)  # creates ones tensor with the same shape as m

# %% [markdown]
"""
### Reshaping a tensor
"""

# %%
m = torch.arange(start=0, end=10, step=1)
m.reshape(2, 5)

# %% [markdown]
"""
### When running into issue with pytorch, ask the following:
1. What is the shape of my tensor?
2. What is the datatype of my tensor?
3. Which device is my tensor on?
"""

# %%
some_tensor = torch.rand(1, 2)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")  # will default to CPU

# %% [markdown]
"""
## Basic operations
"""

# %%
tensor = torch.tensor(data=[1, 2, 3])
tensor + 10

# %%
tensor = torch.tensor(data=[1, 2, 3])
tensor * 10

# %%
tensor = torch.tensor(data=[1, 2, 3])
tensor * 10
tensor  # original tensor is unchanged

# %%
tensor = torch.tensor(data=[1, 2, 3])
torch.mul(input=tensor, other=10)

# %%
tensor = torch.tensor(data=[1, 2, 3])
tensor * 10  # * is shorthand for multiply

# %% [markdown]
"""
### Matrix multiplication (aka dot product)

Two rules:

1. The inner dimensions must match.
    * (3, 2) @ (3, 2) won't work because 2 != 3.
    * (3, 2) @ (2, 4) will work because 2 == 2.
2. Resulting matrix has shape of outer dimensions.
    * (3, 2) @ (2, 4) = (3, 4) aka 3 rows and 4 columns.

@ is shorthand for matrix multiplication.

Element wise multiplication doesn't add the values, opposed to matrix
multiplication.

#### Example:
    t = torch.tensor([1,2,3])

Element wise multiplication:

    t*t = [1*1, 2*2, 3*3]) = [1, 4, 9]
    t.mul(t) = [1*1, 2*2, 3*3]) = [1, 4, 9]

Matrix multiplication:

    t @ t = [1*1 + 2*2 + 3*3] = 14
"""

# %%
tensor = torch.tensor(data=[1, 2, 3])
tensor @ tensor

# %%
t1 = torch.tensor(data=[1, 2])  # shape (2,)
t2 = torch.tensor(data=[[4, 5], [7, 8]])  # shape (2,2)
t1 * t2  # works because of broadcasting

# %%
# Element wise multiplication
t = torch.tensor(data=[1, 2, 3])
t * t == t.mul(t)

# %%
# Matrix multiplication
t = torch.tensor(data=[1, 2, 3])
t @ t == t.matmul(t)

# %% [markdown]
"""
## Transpose
"""

# %%
tensor_A = torch.tensor(data=[[1, 2], [3, 4], [5, 6]])
tensor_B = torch.tensor(data=[[7, 10], [8, 11], [9, 12]])

tensor_A.shape, tensor_B.shape
# tensor_A @ tensor_B # will error

# %%
# transpose makes the inner dimensions match
tensor_A, tensor_B.T

# %%
# transpose makes the inner dimensions match
tensor_A.shape, tensor_B.T.shape

# %%
tensor_A @ tensor_B.T

# %% [markdown]
"""
### Aggregation
"""

# %%
tensor = torch.arange(start=0, end=100, step=10)
tensor

# %%
print("Min:", torch.min(tensor))
print("Max:", torch.max(tensor))
print("Mean:", torch.mean(tensor.float()))  # won't work without float datattype
print("Sum:", torch.sum(tensor))

# %%
# You can call the methods directly on the tensor too
print("Min:", tensor.min())
print("Max:", tensor.max())
print("Mean:", tensor.float().mean())  # won't work without float datattype
print("Sum:", tensor.sum())

# %% [markdown]
"""
### Positional min/max
"""

# %%
tensor = torch.arange(start=0, end=100, step=10)
print("Max index:", torch.argmax(tensor))
print("Min index:", torch.argmin(tensor))

# %% [markdown]
"""
### Changing datatypes
"""

# %%
tensor = torch.arange(start=0, end=100, step=10)
tensor.type(dtype=torch.int8)

# %% [markdown]
"""
### Shapes
"""

# %%
tensor = torch.arange(start=0, end=10, step=1)

# rehape input to given shape if compatible
tensor, torch.reshape(input=tensor, shape=(2, 5))

# %%
# same as reshape, but shares same data as original tensor
tensor = torch.arange(start=0, end=10, step=1)
tensor, tensor.view((2, 5))  # alternative to reshape

# %%
# stack tensor on top of itself 5 times
stacked = torch.stack([tensor, tensor])
stacked

# %%
stacked[0][0] = 2  # only changes the 1st tensor
stacked

# %% [markdown]
"""
### Indexing
"""

# %%
tensor = torch.arange(start=1, end=10).reshape(shape=(1, 3, 3))
print(tensor)
tensor[0][1][2]  # 6

# %%
tensor = torch.arange(start=1, end=10).reshape(shape=(1, 3, 3))

# get all values of 0th dimension and 0th index of 1st dimension
tensor[:, 0]

# %% [markdown]
"""
## Numpy

Two main methods you want to use for Numpy to PyTorch (and back again) are:

    1. torch.from_numpy(ndarray) # converts a numpy array to a tensor
    2. torch.Tensor.numpy()      # converts a tensor to a numpy array

By default Numpy arrays are float64, while PyTorch tensors are float32, so you
may need to change the datatype.
"""
