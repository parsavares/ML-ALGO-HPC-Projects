import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import horovod.torch as hvd
import os
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Initialize Horovod
hvd.init()

# GPU handling with error checking
if torch.cuda.is_available():
    # Ensure we're not trying to access GPUs that don't exist
    n_gpus = torch.cuda.device_count()
    if hvd.local_rank() < n_gpus:
        torch.cuda.set_device(hvd.local_rank())
        device = torch.device(f'cuda:{hvd.local_rank()}')
    else:
        device = torch.device('cpu')
        print(f"Warning: Not enough GPUs. Rank {hvd.local_rank()} using CPU")
else:
    device = torch.device('cpu')
    print("Warning: No GPUs found, using CPU")
    
class CustomTransform:
    def __call__(self, pic):
        img = Image.fromarray(pic)
        img = img.resize((224, 224), Image.BILINEAR)
        return TF.to_tensor(img)

# Load and preprocess CIFAR10 data
transform = CustomTransform()

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Partition dataset
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE // hvd.size(),
    sampler=train_sampler
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE // hvd.size(),
    sampler=test_sampler
)

# Create model
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Move model to appropriate device
model = ResNet50Model().to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * hvd.size())

# Synchronize initial model state
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Wrap optimizer
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    backward_passes_per_step=1
)

criterion = nn.CrossEntropyLoss()

# Add synchronization point
torch.cuda.synchronize() if torch.cuda.is_available() else None

# Training loop
for epoch in range(4):
    model.train()
    train_sampler.set_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    if hvd.rank() == 0:
        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

# Evaluation
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

test_loss /= len(test_loader)
accuracy = correct / total

if hvd.rank() == 0:
    print(f'Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')