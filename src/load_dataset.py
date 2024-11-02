from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_dataset(dataset_name, train=True, transform=None):
    """Loads the appropriate dataset for training/testing"""
    
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root='data/mnist', train=train, download=True, transform=transform) # Updated since it was saving the data outside project directory
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='data/cifar10', train=train, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    
    return dataset


def load_test_loader(dataset_name, num_workers=4, pin_memory=True):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = load_dataset(dataset_name, train=False, transform=transform)

    #testloader = DataLoader(testset, batch_size=100, shuffle=False)
    return DataLoader(
        testset,
        batch_size=128,  # Increase batch size for faster processing
        shuffle=False,
        num_workers=num_workers,  # Use multiple workers for data loading
        pin_memory=pin_memory    # Enable pin memory for faster GPU transfer
    )