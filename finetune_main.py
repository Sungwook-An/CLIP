import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import EfficientNet_ # EfficientNet for No BlurPool
import torch.nn as nn
import torch.optim as optim
import PIL

parser = argparse.ArgumentParser(description='PyTorch ImageNet Finetuning')
parser.add_argument('data', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-ft', '--fine_tuning', default=None, type=str, help='path of the domain specific data (default: none)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('-c', '--classes', default=6, type=int, help='Number of classes')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')

parser.add_argument('--ll', '--lp-learning-rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--lp-momentum', default=0.9, type=float, help='LP momentum')
parser.add_argument('--lp', '--linear-probing', default=10, type=int, help='number of epochs to train the linear probing')

parser.add_argument('--fl', '--ft-learning-rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--ft-momentum', default=0.9, type=float, help='FT momentum')
parser.add_argument('--ft', '--fine-tuning', default=10, type=int, help='number of epochs to train the fine tuning')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True

# Define transformation for training and validation
transform = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = ImageFolder(args.data, transform=transform)
# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Load the pretrained EfficientNet-b0 model
model = EfficientNet_.from_pretrained('efficientnet-b1')

# Modify the final layer to match the number of classes in the dataset
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, args.classes)

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model._fc.parameters(), lr=args.ll, momentum=args.lp_momentum)

# Transfer the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
model = torch.nn.DataParallel(model).to(device)

print("Model loaded successfully!")
print("Linear Probing")
# Linear Probing: Train only the classifier first
for param in model.parameters():
    param.requires_grad = False
for param in model.module._fc.parameters():
    param.requires_grad = True

# Train the classifier
num_epochs = args.lp
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

print("Full Fine-Tuning")
# Full Fine-Tuning: Train all layers
for param in model.parameters():
    param.requires_grad = True

# Adjust the optimizer to include all parameters
optimizer = optim.SGD(model.parameters(), lr=args.fl, momentum=args.ft_momentum)

# Train the entire model
num_epochs = args.ft
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

print("Validation")
# Validate the model
model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Loss: {val_loss/len(val_loader)}")
print(f"Validation Accuracy: {100 * correct / total}%")
