import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model_mnist import MNISTNet
from models.model_cifar10 import CIFAR10Net
from load_dataset import load_dataset
from evaluate_model import evaluate_model, print_cm


def train_reference_model(dataset_name):
    if dataset_name == "cifar10":
        classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classification_labels = [str(i) for i in range(10)]

    device = torch.device('mps')

    train_batch_size = 64
    test_batch_size = 128
    num_epochs = 25

    train_dataset = load_dataset(dataset_name, train=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = load_dataset(dataset_name, train=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    model = MNISTNet() if dataset_name == "mnist" else CIFAR10Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item()}')

        print(f'Epoch: {epoch+1}, Average Loss: {total_loss / len(train_loader)}')
        test_result = evaluate_model(model, test_loader, classification_labels, device=device)
        print(f"Accuracy on {dataset_name}: {test_result['accuracy'] * 100:.2f}%\n")
        print("Classification Report:\n" + test_result['classification_report'])
        print("Confusion Matrix:")
        print_cm(test_result["confusion_matrix"], labels=classification_labels)
        print()

    # save model
    model_save_path = f"models/reference_{dataset_name}"
    torch.save(model.state_dict, model_save_path)
    print(f"Model saved in {model_save_path}")


if __name__ == "__main__":
    dataset_name = "mnist"
    # dataset_name = "cifar10"
    train_reference_model(dataset_name)
