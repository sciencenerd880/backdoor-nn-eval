import torch
import matplotlib.pyplot as plt

from load_model import load_model
from load_dataset import load_dataset



if __name__ == "__main__":
    model_name = "model1"
    dataset_name = "mnist"
    target_label = 7

    model, device = load_model(model_name, dataset_name)
    trigger_file = "output/neural_cleanse_experiment_model1/triggers/suspicious/trigger_mask_model1_class_7.pt"

    trigger = torch.load(trigger_file)

    dataset = load_dataset(dataset_name, train=False)

    used_image = None

    for image, label in dataset:
        if label != target_label:
            used_image = image
            break

    sigmoid_mask = torch.sigmoid(trigger["mask"])
    tanh_delta = torch.tanh(trigger["delta"])
    triggered_image = used_image * (1 - sigmoid_mask) + sigmoid_mask * tanh_delta

    plt.subplot(1, 2, 1)
    plt.title("Clean Image")
    plt.imshow(used_image.permute(1, 2, 0))

    plt.subplot(1, 2, 2)
    plt.title("Perturbed Image")
    plt.imshow(triggered_image.permute(1, 2, 0))
    plt.savefig(f"output/{model_name}_{dataset_name}.png")