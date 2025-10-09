from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
import pandas as pd

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
# (1) Randomly pick some samples and view individual images and corresponding labels from the dataset
num_samples = 10
random_indices = np.random.choice(len(train_images), num_samples, replace=False)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[idx].cpu().numpy(), cmap='gray')
    plt.title(f'Label: {train_labels[idx]}')
    plt.axis('off')
plt.suptitle('Randomly Selected Samples')
plt.show()

# save
plt.savefig('results/mnist_random_samples.png')
plt.close()

# (2) Analyze the distribution of digits in the dataset
plt.figure(figsize=(10, 5))

# 统计训练标签中每个数字的出现次数
# count frenquecy of each number in training dataset
unique, counts = np.unique(train_labels.numpy(), return_counts=True)  # Covert to NumPy
digit_distribution = dict(zip(unique, counts))

plt.bar(digit_distribution.keys(), digit_distribution.values(), color='skyblue')
plt.xlabel('Digits')
plt.ylabel('Number of Samples')
plt.title('Distribution of Digits in the Training Dataset')
plt.xticks(range(10))  
plt.grid(axis='y')
plt.show()

# save
plt.savefig('results/mnist_digit_distribution.png')
plt.close()

# Generate statistical summaries of the dataset
# ensure float
train_images = train_images.float() / 255.0 # norm

# flat image
train_images_flat = train_images.view(train_images.size(0), -1)  # flat image

mean = train_images_flat.mean(dim=1)
std = train_images_flat.std(dim=1)

# visualize
summary_df = pd.DataFrame({
    'Digit': unique,
    'Count': counts,
    'Mean Pixel Value': [train_images_flat[train_labels == digit].mean().item() for digit in unique],
    'Std Pixel Value': [train_images_flat[train_labels == digit].std().item() for digit in unique]
})

print(summary_df)

# save as csv files
summary_df.to_csv('results/mnist_dataset_summary.csv', index=False)


# Task 1: define the model
# Task 1.1: Implement a Multi-Layer Perceptron (MLP) with 2 hidden layers
class MLP(nn.Module):
    def __init__(self, num_of_neurons_in_hidden_layer=256, num_of_classes=10):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(28*28, num_of_neurons_in_hidden_layer)
        # your_code_here
        self.hidden_layer = nn.Linear(num_of_neurons_in_hidden_layer, num_of_neurons_in_hidden_layer)
        self.output_layer = nn.Linear(num_of_neurons_in_hidden_layer, num_of_classes)

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), '\
            Input shape should be (batch_size, 1, 28, 28)'
        x = x.view(-1, 28*28)
        # your_code_here
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    
# Task 1.2: Implement a Vanilla Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self, num_of_classes=10):
        super(CNN, self).__init__()
        # your_code_here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_of_classes)

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), '\
            Input shape should be (batch_size, 1, 28, 28)'
        # your_code_here
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Task 1.3: Implement a LeNet5 Neural Network (LeNet5)
class LeNet5(nn.Module):
    def __init__(self, num_of_classes=10):
        super(LeNet5, self).__init__()
        # your_code_here
        # 2 Conv layer and 3 Linear layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_of_classes)

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[1:] == (1, 28, 28), '\
            Input shape should be (batch_size, 1, 28, 28)'
        # your_code_here
        # conv -> act -> pool
        x = self.conv1(x)
        x = F.tanh(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = F.tanh(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
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
# criterion = None
criterion = nn.CrossEntropyLoss()

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