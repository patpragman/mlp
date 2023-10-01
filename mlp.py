import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model
from sklearn.metrics import classification_report
import yaml
from pprint import pprint
from pathlib import Path
from hashlib import sha256
import os

HOME_DIRECTORY = Path.home()
SEED = 42


# Define the architecture of the MLP for image classification
class MLPImageClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 num_classes, activation_function="relu"):
        super(MLPImageClassifier, self).__init__()
        layers = []

        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        if activation_function.lower() == "relu":
            fn = nn.ReLU()
        elif activation_function.lower() == "leaky_relu":
            fn = nn.LeakyReLU(0.1)
        else:
            fn = nn.Tanh()

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(fn)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (assuming images as input)
        x = self.mlp(x)
        return x





with open("mlp_sweep.yml", "r") as yaml_file:
    sweep_config = yaml.safe_load(yaml_file)

sweep_id = wandb.sweep(sweep=sweep_config)
def find_best_model():
    # config for wandb

    wandb.init(project='Elodea MLP')
    # Initialize wandb
    config = wandb.config

    # creating the model stuff
    input_size = 3*config.input_size**2
    hidden_sizes = [config.hidden_sizes for i in range(0, config.hidden_depth)]
    num_classes = 2  # this doesn't ever change
    learning_rate = config.learning_rate
    epochs = wandb.config.epochs

    # Create the MLP-based image classifier model
    model = MLPImageClassifier(input_size,
                               hidden_sizes,
                               num_classes, activation_function=config.activation_function)

    path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"

    dataset = FloatImageDataset(directory_path=path,
                                true_folder_name="entangled", false_folder_name="not_entangled"
                                )

    training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
    batch_size = config.batch_size

    # create the dataloaders
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimzer parsing logic:
    if config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                   model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs,
                                   device="cpu", wandb=wandb, verbose=False)

    # save the model
    model_hash = sha256(str(config).encode('utf-8'))
    model_name = f"alexnet_{model_hash.hexdigest()}"
    if not os.path.isdir(f"models/{model_name}"):
        os.mkdir(f"models/{model_name}")

    y_true, y_pred = history['y_true'], history['y_pred']
    cr = classification_report(y_true=y_true, y_pred=y_pred)

    report = [
        model_name, cr, str(model)
    ]
    with open(f"models/{model_name}/report.md", "w") as report_file:
        report_file.writelines(report)

    torch.save(model.state_dict(), f"models/{model_name}.pth")

    # Log hyperparameters to wandb
    wandb.log(dict(config))

if __name__ == "__main__":
    wandb.agent(sweep_id, function=find_best_model)