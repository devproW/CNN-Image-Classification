import os
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter, writer
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import wget
import zipfile


# Build CNN model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        num_input = 3  # channels
        num_output = 3
        self.conv1 = nn.Conv2d(in_channels=num_input, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        # Note that the input of this layers is depending on your input image sizes
        self.fc1 = nn.Linear(in_features=self.linear_input_neurons(), out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # here we apply convolution operations before linear layer, and it returns the 4-dimensional size tensor.
    def size_after_relu(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))

        return x.size()
        # after obtaining the size in above method, we call it and multiply all elements of the returned size.

    def linear_input_neurons(self):
        size = self.size_after_relu(torch.rand(1, 3, 150, 150))  # image size: 150x150, channel: 3
        m = 1
        for i in size:
            m *= i

        return int(m)


def train(model, loader, optimizer, criterion, saving_path, n_epochs_stop, epochs):
    epochs_no_improve = 0
    min_val_loss = np.Inf
    early_stop = False

    for i in range(1, epochs + 1):

        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_size = 0.0
            epoch_loss = 0.0
            correct = 0

            if phase == 'train':
                model.train()
            else:
                # To set dropout and batch normalization layers to evaluation mode before running inference
                model.eval()

            for images, labels in loader[phase]:

                with torch.set_grad_enabled(phase == 'train'):
                    # Predict and compute loss
                    y_pred = model(images)
                    loss = criterion(y_pred, labels)

                    # Backpropogate and update parameters
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Note: the last batch size will be the remainder of data size divided by batch size
                # Thus, we multiply one batch loss with the number of items in batch and divide it by total size later
                # The final running size is equal to the data size
                running_loss += loss.item() * y_pred.size(0)
                running_size += y_pred.size(0)

                # The predictions is the index of the maximum values in the output of model
                predictions = torch.max(y_pred, 1)[1]
                correct += (predictions == labels).sum().item()

            epoch_loss = running_loss / running_size
            epoch_accuracy = correct / running_size
            writer.add_scalars('Loss', {phase: epoch_loss}, i)
            writer.add_scalars('Accuracy', {phase: epoch_accuracy}, i)

            # Print score at every epoch
            if (i % 1 == 0 or i == 1):
                if phase == 'train':
                    print(f'Epoch {i}:')
                print(f'  {phase.upper()} Loss: {epoch_loss}')
                print(f'  {phase.upper()} Accuracy: {epoch_accuracy}')

            # For visualization (Optional)
            loss_score[phase].append(epoch_loss)

            # Early stopping
            if phase == 'valid':
                if epoch_loss < min_val_loss:
                    # Save the model before the epochs start to not improving
                    torch.save(model.state_dict(), saving_path)
                    print(f"Model saved at Epoch {i} \n")
                    epochs_no_improve = 0
                    min_val_loss = epoch_loss

                else:
                    epochs_no_improve += 1
                    print('\tepochs_no_improve:',
                          epochs_no_improve, ' at Epoch', i)

            if epochs_no_improve == n_epochs_stop:
                print('\nEarly stopping!')
                early_stop = True
                break

        # To exit loop
        if early_stop:
            print("Stopped")
            break

    writer.close()


if __name__ == "__main__":
    # The data is located in the resources/data folder
    datadir = 'resources/data/fruits_image_classification'
    traindir = datadir + '/train/'
    validdir = datadir + '/validation/'
    testdir = datadir + '/test/'
    dirtytestdir = datadir + '/dirty_test/'

    # Check our images number in the train, val and test folders (Optional)
    # Iterate through each category
    categories = []
    train_size, val_size, test_size, dirtytest_size = 0, 0, 0, 0

    for category in os.listdir(traindir):
        categories.append(category)

        # Number of images added up
        train_imgs = os.listdir(Path(traindir) / f'{category}')
        valid_imgs = os.listdir(Path(validdir) / f'{category}')
        test_imgs = os.listdir(Path(testdir) / f'{category}')
        dirtytest_imgs = os.listdir(Path(dirtytestdir) / f'{category}')
        train_size += len(train_imgs)
        val_size += len(valid_imgs)
        test_size += len(test_imgs)
        dirtytest_size += len(dirtytest_imgs)

    print(f'Train set: {train_size}, Validation set: {val_size}, Test set:{test_size}, Dirty test set:{dirtytest_size}',
          end='\n\n')
    print(categories)
    print(f'\nNumber of categories: {len(categories)}')

    # We will need our input in tensors form, must `transforms.ToTensor(),`

    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                # You can set 'Resize' and 'Crop' to higher resolution for better result
                transforms.Resize(180),

                # Data augmented here
                # Use (224, 244) if you want to train on Imagenet pre-trained model
                transforms.RandomCrop(150),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
            ]),

        # Validation and Inference do not use augmentation
        'valid':
            transforms.Compose([
                # You can set to higher resolution for better result
                transforms.Resize(150),
                transforms.CenterCrop(150),
                transforms.ToTensor(),
            ]),
    }

    torch.manual_seed(123)

    # Datasets from folders
    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
            datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
    }

    # Dataloader iterators, make sure to shuffle
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=10, shuffle=True),
        'valid': DataLoader(data['valid'], batch_size=10, shuffle=True)
    }

    # To check the iterative behavior of the DataLoader (optional)
    # Iterate through the dataloader once
    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    print(features.shape)
    print(labels.shape)

    # Visualization of images in dataloader (optional)
    # nrow = Number of images displayed in each row of the grid.
    # Clip image pixel value to 0-1
    grid = torchvision.utils.make_grid(np.clip(features[0:10], 0, 1), nrow=10)

    plt.figure(figsize=(15, 15))
    # Transpose to show in rows / horizontally
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    print("Labels: ")
    print(labels[0:10])
    for i in labels[0:10]:
        print(categories[i] + ", ", end="")

    # Define model, dataloaders, optimizer, criterion
    model = Net()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=1e-4, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    # Setup writer for tensorboard and also a saving_path to save model (Optional)
    writer = SummaryWriter('logs/fruits_classifier')
    if not os.path.exists('resources/model'):
        os.mkdir('resources/model')
    saving_path = 'resources/model/fruit_classifier_state_dict.pt'

    n_epochs_stop = 5
    epochs = 50
    loss_score = {'train': [], 'valid': []}

    train(model, dataloaders, optimizer, criterion,
          saving_path, n_epochs_stop, epochs)

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)
    ax.set_title("Loss Score against Epoch")
    ax.grid(visible=True)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Loss Score")
    ax.plot(loss_score['train'], color='red', label='Training Loss')
    ax.plot(loss_score['valid'], color='green', label='Validation Loss')
    ax.legend()
    plt.show()
