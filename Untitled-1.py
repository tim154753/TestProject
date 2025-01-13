import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.softmax(self.output(x), dim = 1)
        return x

def load_images(file_path = R'C:\Users\timma\Downloads\MNIST\train-images.idx3-ubyte'):
    with open(file_path, 'rb') as file:
        file.read(16)
        data = np.fromfile(file, dtype=np.uint8)
        return data.reshape(-1, 784)
    
def load_labels(file_path = R'c:\Users\timma\Downloads\MNIST\train-labels.idx1-ubyte'):
    with open(file_path, 'rb') as file:
        file.read(8)
        labels = np.fromfile(file, dtype=np.uint8)
        return labels

train_images = torch.tensor(load_images(), dtype = torch.float32)
train_labels = torch.tensor(load_labels(), dtype = torch.long)

train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

model = NeuralNetwork(784, 100, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

num_epochs = 10

for epoch in range(num_epochs):
    for batch_images, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()