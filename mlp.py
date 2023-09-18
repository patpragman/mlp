import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model
import pandas as pd
import matplotlib.pyplot as plt

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
        print(input_size, hidden_sizes, num_classes)
        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        layers.append(nn.Softmax())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (assuming images as input)
        return self.mlp(x)


# Define hyperparameters as config
config = dict(
    input_size=3*224**2,  # 3 channels (RGB) and various sized images
    hidden_sizes=[512, 256, 128, 64, 32, 16],  # You can tune the number of neurons in hidden layers
    num_classes=2,  # Number of classes for image classification - for us, 2
    learning_rate=0.0001,  # You can tune the learning rate
    epochs=50,  # You can tune the number of epochs
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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                               model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=150,
                               device="cpu", wandb=wandb, verbose=False)


# Log test accuracy to wandb
wandb.log(history)

df = pd.DataFrame(history)
# Set "epoch" as the x-axis for all columns except "epoch"
df.set_index("epoch", inplace=True)

# Plot all columns
df.plot(marker='o', linestyle='-')

# Customize the plot
plt.xlabel("Epoch")
plt.ylabel("Values")
plt.title("Line Plot for DataFrame Columns")
plt.legend(title="Columns")
plt.show()

# Log hyperparameters to wandb
wandb.log(config)
