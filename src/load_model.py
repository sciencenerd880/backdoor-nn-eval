import torch
import torch.nn as nn
import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_mnist import MNISTNet  
from models.model_cifar10 import CIFAR10Net  

def load_model(model_name, dataset):
    """Loads the pre-trained model from the provided name and dataset"""
    
    # Detect if CUDA is available
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # Forcing CPU-only mode
    #print(f"Using device: {device}")
    
    if dataset == 'mnist':
        model = MNISTNet()
    elif dataset == 'cifar10':
        model = CIFAR10Net()
    else:
        raise ValueError("Unsupported dataset")
    
    # Explicitly pointing to 'mnist_bd.pt' for MNIST dataset otherwise find the others based on CIFAR10 or 100
    model_path = f'models/{model_name}/mnist_bd.pt' if dataset == 'mnist' else f'models/{model_name}/{dataset}_bd.pt'
    
    # Ensure the path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' does not exist.")
    
    model.load_state_dict(torch.load(model_path))
    
    return model.to(device), device
