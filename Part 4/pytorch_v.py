import argparse
import os
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import datasets, transforms
import horovod
import horovod.torch as hvd
import time
import warnings
import torch

warnings.filterwarnings("ignore")

BATCH_SIZE = 256
LEARNING_RATE = 0.001

torch.manual_seed(42)

# Horovod initialize
torch.cuda.empty_cache()
hvd.init()

# Adjust batch size for distributed training
local_batch_size = BATCH_SIZE // int(hvd.size())

# GPU setup
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(42)
    kwargs = {"num_workers": 1, "pin_memory": True}
else:
    kwargs = {}

torch.set_num_threads(1)

def get_cifar10_dataset(is_training, local_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    with FileLock(os.path.expanduser("~/data.lock")):
        torch_dataset = datasets.CIFAR10(
            "./data/",
            train=is_training,
            download=True,
            transform=transform
        )

    torch_sampler = torch.utils.data.distributed.DistributedSampler(
        torch_dataset, 
        num_replicas=hvd.size(), 
        rank=hvd.rank(),
        shuffle=is_training
    )
    
    torch_loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=local_batch_size,
        sampler=torch_sampler,
        drop_last=True,
        **kwargs
    )
    
    return torch_loader

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training function
def train_epoch(epoch, train_loader, optimizer, model):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            # Clear cache before loading new batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                data, target = data.cuda(), target.cuda()
            
            # Add batch size check
            if data.size(0) != local_batch_size:
                print(f"Skipping batch with incorrect size: {data.size(0)}")
                continue
                
            optimizer.zero_grad()
            output = model(data)
            
            if output.size(0) != target.size(0):
                print(f"Size mismatch: output {output.size(0)}, target {target.size(0)}")
                continue
                
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
        except RuntimeError as e:
            print(f"CUDA error in batch {batch_idx}: {str(e)}")
            torch.cuda.empty_cache()
            continue

# Metric averaging function
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

# Testing function
def test(test_loader, model):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.sampler)
    test_accuracy /= len(test_loader.sampler)

    loss = metric_average(test_loss, "avg_loss")
    accuracy = metric_average(test_accuracy, "avg_accuracy")

    return loss, accuracy

# Main execution
if __name__ == "__main__":
    # Create data loaders
    train_loader = get_cifar10_dataset(True, local_batch_size)
    test_loader = get_cifar10_dataset(False, local_batch_size)

    # Create model
    model = Net()
    if torch.cuda.is_available():
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs
    optimizer = optim.SGD(model.parameters(), 
                         lr=LEARNING_RATE * hvd.size(),
                         momentum=0.9)

    # Horovod: broadcast parameters & optimizer state
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    # Training loop
    hvd.allgather(torch.tensor([0]))
    start_time = time.time()
    
    for epoch in range(4):
        train_epoch(epoch, train_loader, optimizer, model)
    
    print("Elapsed training time: ", round(time.time() - start_time), " sec")

    # Final test
    loss, accuracy = test(test_loader, model)
    if hvd.rank() == 0:
        print(f"Test set: Average loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
