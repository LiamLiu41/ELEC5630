from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

# Set the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the training and test data
train_data = MNIST(root='data', train=True, download=True)
test_data = MNIST(root='data', train=False, download=True)
# Extract the images and labels
train_images, train_labels = train_data.data, train_data.targets
test_images, test_labels = test_data.data, test_data.targets
# Print the shape of the data
print('Train images shape:', train_images.shape, 'Train labels shape:', train_labels.shape)
print('Test images shape:', test_images.shape, 'Test labels shape:', test_labels.shape)

# Task 0: Do some data visualization and preprocessing here
## your_code_here


# Task 1: define the model
# Task 1.1: Implement a Multi-Layer Perceptron (MLP) with 2 hidden layers
class MLP(nn.Module):
    def __init__(self, num_of_neurons_in_hidden_layer=256, num_of_classes=10):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(28*28, num_of_neurons_in_hidden_layer)
        # your_code_here

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), '\
            Input shape should be (batch_size, 1, 28, 28)'
        x = x.view(-1, 28*28)
        # your_code_here
        return x
    
# Task 1.2: Implement a Vanilla Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self, num_of_classes=10):
        super(CNN, self).__init__()
        # your_code_here

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), '\
            Input shape should be (batch_size, 1, 28, 28)'
        # your_code_here
        return x

# Task 1.3: Implement a LeNet5 Neural Network (LeNet5)
class LeNet5(nn.Module):
    def __init__(self, num_of_classes=10):
        super(LeNet5, self).__init__()
        # your_code_here

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), '\
            Input shape should be (batch_size, 1, 28, 28)'
        # your_code_here
        return x

models = {
    'MLP': MLP(),
    'CNN': CNN(),
    'LeNet5': LeNet5()
}

# define training parameters
epochs = 3
batch_size = 64
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Task 2: implement the CrossEntropyLoss here and compare with the official torch API (opt. let's try L2 loss?)
criterion = None

accuracy = []
for model_type, model in models.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # try different optimizers?
    print(f'Training {model_type} model on {device} using {epochs} epochs, {batch_size} batch size and {learning_rate} learning rate')
    # train the model
    cur_accuracy = []
    for epoch in tqdm(range(epochs)):
        print('Epoch:', epoch)
        for i in range(0, len(train_images), batch_size):
            # Extract the batch
            batch_images = train_images[i:i+batch_size].to(device)
            batch_labels = train_labels[i:i+batch_size].to(device)
            # Forward pass
            outputs = model(batch_images.unsqueeze(1).float())
            if criterion is None:
                raise NotImplementedError('Please Implement the CrossEntropyLoss in Task 2')
            loss = criterion(outputs, batch_labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Test the model
        predictions = model(test_images.unsqueeze(1).float().to(device))
        predictions = torch.argmax(predictions, dim=1)
        # Calculate the accuracy
        cur_accuracy.append(np.mean(predictions.cpu().numpy() == test_labels.cpu().numpy()))
        print('Accuracy:', cur_accuracy[-1])

    # Plot the current accuracy curve
    plt.plot(range(1, epochs+1), cur_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_type} Accuracy')
    plt.savefig(f'{model_type}_Accuracy.png')
    plt.clf()
    accuracy.append(cur_accuracy[-1])

# Plot the accuracy
plt.bar(range(1, len(accuracy)+1), accuracy)
plt.xticks(range(1, len(accuracy)+1), ['MLP', 'CNN', 'LeNet5'])
for a,b in zip(range(1, len(accuracy)+1), accuracy):
    plt.text(a,b,
            b,
            ha='center', 
            va='bottom',
            )
plt.xlabel('Number of neurons in hidden layer')
plt.ylabel('Accuracy')
plt.title('Accuracy of different types of NNs')
plt.savefig('Accuracy.png')