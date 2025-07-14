import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## Training
# Initialize parameters
W1 = torch.randn(784, 500) / np.sqrt(784)  # first layer
W1.requires_grad_()
b1 = torch.zeros(500, requires_grad=True)

W2 = torch.randn(500, 10) / np.sqrt(500)  # second layer
W2.requires_grad_()
b2 = torch.zeros(10, requires_grad=True)

# Optimizer
optimizer = torch.optim.SGD([W1, b1, W2, b2], lr=0.1)

# Iterate through train set minibatchs
for images, labels in tqdm(train_loader):
    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass
    x = images.view(-1, 28 * 28)
    h = torch.matmul(x, W1) + b1  # First transformation (hidden layer)
    h = F.relu(h)  # Apply ReLU activation
    y = torch.matmul(h, W2) + b2

    cross_entropy = F.cross_entropy(y, labels)

    # Backward pass
    cross_entropy.backward()
    optimizer.step()


## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, 28 * 28)
        z = torch.matmul(x, W1) + b1
        z = F.relu(z)
        y = torch.matmul(z, W2) + b2

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct / total))

# After training is done, save the model's parameters (weights and biases)
#torch.save({'W': W, 'b': b}, 'mnist_model.pth')
#This will save the model's parameters in a file named 'mnist_model.pth'.



#Test accuracy: 0.927299976348877