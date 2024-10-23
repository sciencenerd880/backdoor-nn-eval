import torch
from torch.utils.data import DataLoader
from load_model import load_model
from load_dataset import load_dataset

# Adapted from the lab exercise
def evaluate_model(model, dataset):
    """Evaluates model accuracy on clean test data"""
    
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    print(f'Accuracy on clean test data: {accuracy * 100:.2f}%')
    
    return accuracy

if __name__ == "__main__":
    # Choose model and dataset
    model_name = 'model1'  # As per the models' subfolder, Change to your desired model name (e.g., 'model1', 'model2')
    dataset_name = 'mnist'  # Change to the desired dataset ('mnist' or 'cifar10')
    
    model = load_model(model_name, dataset_name)
    dataset = load_dataset(dataset_name, train=False)
    
    evaluate_model(model, dataset)
