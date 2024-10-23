import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from load_model import load_model
from load_dataset import load_dataset


# Adapted from the lab exercise
def evaluate_model(model, dataset, classification_labels):
    """Evaluates model accuracy on clean test data"""
    
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    labels = []
    preds = []
    
    with torch.no_grad():
        for data, target in test_loader:
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
    
    model_name = 'model2'  # As per the models' subfolder, Change to your desired model name (e.g., 'model1', 'model2')
    dataset_name = 'cifar10'  # Change to the desired dataset ('mnist' or 'cifar10')
    if dataset_name == "cifar10":
        classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classification_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    model, device = load_model(model_name, dataset_name)
    dataset = load_dataset(dataset_name, train=False)
    
    result = evaluate_model(model, dataset, classification_labels)
    print(f"Accuracy for {model_name} on {dataset_name}: {result['accuracy'] * 100:.2f}%\n")
    print("Classification Report:\n" + result['classification_report'])
    # sns.heatmap(result["confusion_matrix"], annot=True, yticklabels=classification_labels, xticklabels=classification_labels)
    # plt.show()
    print("Confusion Matrix:")
    print_cm(result["confusion_matrix"], labels=classification_labels)
    
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
        if dataset_name == "cifar10":
            classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            classification_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        print(f"Evaluating {model_name} using {dataset_name} dataset..")
        
        model, device = load_model(model_name, dataset_name)
        dataset = load_dataset(dataset_name, train=False)
        
        result = evaluate_model(model, dataset, classification_labels)
        print(f"Accuracy for {model_name} on {dataset_name}: {result['accuracy'] * 100:.2f}%\n")
        print("Classification Report:\n" + result['classification_report'])
        print("Confusion Matrix:")
        print_cm(result["confusion_matrix"], labels=classification_labels)
'''