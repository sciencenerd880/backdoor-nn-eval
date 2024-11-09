import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from load_model import load_model
from load_dataset import load_dataset


# Adapted from the lab exercise
def evaluate_model(model, test_loader, classification_labels, device):
    """Evaluates model accuracy on clean test data"""
    
    model.eval()
    correct = 0
    total = 0
    labels = []
    preds = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            labels.extend(target.tolist())
            preds.extend(predicted.tolist())
    
    accuracy = correct / total
    report = classification_report(labels, preds, target_names=classification_labels)
    conf_mat = confusion_matrix(labels, preds)
    #print(f'Accuracy on clean test data: {accuracy * 100:.2f}%')
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_mat,
    }


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    # pretty print for confusion matrixes
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}s".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


if __name__ == "__main__":
    # Note so far, model evaluation for model1 mnist works, but cifar10 has some cuda issues... 

    models = [
        ("reference_mnist", "mnist"),
        ("reference_cifar10", "cifar10"),
        ("model1", "mnist"),
        ("model2", "cifar10"),
        ("model3", "cifar10"),
        ("model4", "cifar10"),
        ("model5", "cifar10"),
    ]
    
    for model_name, dataset_name in models:
        if dataset_name == "cifar10":
            classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            classification_labels = [str(i) for i in range(10)]
        model, device = load_model(model_name, dataset_name)
        dataset = load_dataset(dataset_name, train=False)
        
        test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        result = evaluate_model(model, test_loader, classification_labels, device)
        print()
        print(f"Accuracy for {model_name} on {dataset_name}: {result['accuracy'] * 100:.2f}%\n")
        print("Classification Report:\n" + result['classification_report'])
        # sns.heatmap(result["confusion_matrix"], annot=True, yticklabels=classification_labels, xticklabels=classification_labels)
        # plt.show()
        print("Confusion Matrix:")
        print_cm(result["confusion_matrix"], labels=classification_labels)
