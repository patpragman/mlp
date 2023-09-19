import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Initialize wandb
wandb.init(project="Elodea MLP")


# Define the architecture of the MLP for image classification
class MLPImageClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 num_classes):
        super(MLPImageClassifier, self).__init__()
        layers = []

        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.LeakyReLU(0.1))


        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (assuming images as input)
        x = self.mlp(x)
        return x


# Define hyperparameters as config
config = dict(
    input_size=3*224**2,  # 3 channels (RGB) and various sized images
    hidden_sizes=[4096, 4096, 4096],  # You can tune the number of neurons in hidden layers
    num_classes=2,  # Number of classes for image classification - for us, 2
    learning_rate=0.00001,  # You can tune the learning rate
    epochs=75,  # You can tune the number of epochs
)

# Initialize wandb config
wandb.config.update(config)

# creating the model stuff
input_size = wandb.config.input_size
hidden_sizes = wandb.config.hidden_sizes
num_classes = wandb.config.num_classes
learning_rate = wandb.config.learning_rate
epochs = wandb.config.epochs

# Create the MLP-based image classifier model
model = MLPImageClassifier(input_size, hidden_sizes, num_classes)

# Print the architecture of the model
print(model)

path = "/home/patrickpragman/PycharmProjects/models/data_manufacturer/0.35_reduced_then_balanced/data_224"
dataset = FloatImageDataset(directory_path=path,
                            true_folder_name="entangled", false_folder_name="not_entangled"
                            )

training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=42)
batch_size = 32

# create the dataloaders
train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs,
                               device="cpu", wandb=wandb, verbose=False)

y_true, y_pred = history['y_true'], history['y_pred']
print(y_true, y_pred)

print(classification_report(y_true=y_true, y_pred=y_pred))

# Log test accuracy to wandb
wandb.log(history)



# Log hyperparameters to wandb
wandb.log(config)
