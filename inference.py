import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms


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


# To load a saved state dict, we need to instantiate the model first
best_model = Net()

# Notice that the load_state_dict() function takes a dictionary object, NOT a path to a saved object.
# This means that you must deserialize the saved state_dict before you pass it to the load_state_dict() function.
best_model.load_state_dict(torch.load(
    'resources/model/fruit_classifier_state_dict.pt'))
best_model.eval()

# The data is located in the resources/data folder
datadir = 'resources/data/fruits_image_classification'
testdir = datadir + '/test/'
dirtytestdir = datadir + '/dirty_test/'

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

test_data = datasets.ImageFolder(
    root=testdir, transform=image_transforms['valid'])
testloader = DataLoader(test_data, len(test_data), shuffle=False)

with torch.no_grad():
    for images, labels in testloader:
        correct = 0
        y_pred = best_model(images)
        predictions = torch.max(y_pred, 1)[1]
        print(predictions)
        print(labels)
        correct += (predictions == labels).sum().item()
        accuracy = correct / len(test_data)
        print(f"Test Accuracy: {accuracy}")
