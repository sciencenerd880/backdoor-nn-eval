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
    #print(f'Accuracy on clean test data: {accuracy * 100:.2f}%')
    
    return accuracy

if __name__ == "__main__":
    # Note so far, model evaluation for model1 mnist works, but cifar10 has some cuda issues... 
    
    model_name = 'model1'  # As per the models' subfolder, Change to your desired model name (e.g., 'model1', 'model2')
    dataset_name = 'mnist'  # Change to the desired dataset ('mnist' or 'cifar10')
    model, device = load_model(model_name, dataset_name)
    dataset = load_dataset(dataset_name, train=False)
    
    accuracy = evaluate_model(model, dataset)
    print(f"Accuracy for {model_name} on {dataset_name}: {accuracy * 100:.2f}%\n")
    
'''
if __name__ == "__main__":
    # Choose model and dataset
    #model_name = 'model1'  # As per the models' subfolder, Change to your desired model name (e.g., 'model1', 'model2')
    #dataset_name = 'mnist'  # Change to the desired dataset ('mnist' or 'cifar10')
    
    model_dataset_mapping = {
        'model1': 'mnist',   # Model 1 is trained on MNIST
        'model2': 'cifar10',  # The rest are trained on CIFAR-10
        'model3': 'cifar10',
        'model4': 'cifar10',
        'model5': 'cifar10'
    }
        
    for model_name, dataset_name in model_dataset_mapping.items():
        print(f"Evaluating {model_name} using {dataset_name} dataset..")
        
        model, device = load_model(model_name, dataset_name)
        dataset = load_dataset(dataset_name, train=False)
        
        accuracy = evaluate_model(model, dataset)
        print(f"Accuracy for {model_name} on {dataset_name}: {accuracy * 100:.2f}%\n")
    
'''