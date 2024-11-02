import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_dataset import load_dataset
from evaluate_model import evaluate_model, print_cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_mnist import MNISTNet
from models.model_cifar10 import CIFAR10Net


def train_reference_model(dataset_name):
    if dataset_name == "cifar10":
        classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classification_labels = [str(i) for i in range(10)]

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps')

    train_batch_size = 64
    test_batch_size = 128
    num_epochs = 25

    train_dataset = load_dataset(dataset_name, train=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = load_dataset(dataset_name, train=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    model = MNISTNet() if dataset_name == "mnist" else CIFAR10Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
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
    model_save_path = f"models/reference_{dataset_name}/{dataset_name}_reference.pt"
    torch.save(model.state_dict, model_save_path)
    print(f"Model saved in {model_save_path}")


if __name__ == "__main__":
    # dataset_name = "mnist"
    dataset_name = "cifar10"
    train_reference_model(dataset_name)
