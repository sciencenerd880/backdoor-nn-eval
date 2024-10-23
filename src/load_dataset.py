from torchvision import datasets, transforms

def load_dataset(dataset_name, train=True):
    """Loads the appropriate dataset for training/testing"""
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root='../data/mnist', train=train, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='../data/cifar10', train=train, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    
    return dataset
