import torch
from torch.utils.data import DataLoader
from load_model import load_model
from load_dataset import load_dataset
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model_name = "model2"
    dataset_name = "cifar10"
    ACTUAL_CLASS = 1
    PREDICTED_CLASS = 6

    dataset = load_dataset(dataset_name, train=False)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    if dataset_name == "cifar10":
        classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classification_labels = [str(i) for i in range(10)]

    model, device = load_model(model_name, dataset_name)
    dataset = load_dataset(dataset_name, train=False)
    model.eval()
    correct = 0
    total = 0
    labels = []
    preds = []

    idx = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            labels.extend(target.tolist())
            preds.extend(predicted.tolist())

            suspicious_data = data[(target == ACTUAL_CLASS) & (predicted == PREDICTED_CLASS)]
            # save every suspicious_data
            for each_sus in suspicious_data:
                image_plot_ready = each_sus.detach().cpu().numpy().transpose(1, 2, 0)
                plt.figure(figsize=(4,4))
                plt.imshow(image_plot_ready)
                plt.title(f"True Label: {classification_labels[ACTUAL_CLASS]}, Predicted: {classification_labels[PREDICTED_CLASS]}")
                plt.savefig(f"output/wrong/{model_name}_{classification_labels[ACTUAL_CLASS]}_predicted_{classification_labels[PREDICTED_CLASS]}_{idx}.png")
                idx += 1
                plt.close()

    accuracy = correct / total

    print(accuracy)