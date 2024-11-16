import random
import torch
import matplotlib.pyplot as plt
from load_model import load_model
from load_dataset import load_dataset


if __name__ == "__main__":
    model_name = "model4"
    dataset_name = "cifar10"
    # dataset_name = "mnist"
    target_label = 6

    model, device = load_model(model_name, dataset_name)
    # trigger_file = "output/neural_cleanse_experiment_model1/triggers/suspicious/trigger_mask_model1_class_7.pt"
    trigger_file = "output/neural_cleanse_experiment_model4/triggers/all/trigger_mask_model4_class_dog.pt"
    # trigger_file = "output/neural_cleanse_experiment_model5/triggers/suspicious/trigger_mask_model5_class_truck.pt"
    trigger = torch.load(trigger_file)

    classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'] if dataset_name == "cifar10" else list(range(10))

    dataset = load_dataset(dataset_name, train=False)
    shuffled_dataset = sorted(dataset, key=lambda x: random.random())

    clean_image = None

    for image, label in shuffled_dataset:
        if label != target_label:
            clean_image = image
            break

    sigmoid_mask = torch.sigmoid(trigger["mask"])
    tanh_delta = torch.tanh(trigger["delta"])
    triggered_image = clean_image * (1 - sigmoid_mask) + sigmoid_mask * tanh_delta

    data_to_predict = torch.stack([clean_image, triggered_image], dim=0).to(device)
    prediction = model(data_to_predict)
    predicted_class = torch.argmax(prediction, dim=1).detach().cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.title("Clean Image")
    plt.imshow(clean_image.permute(1, 2, 0).clip(0, 1))
    plt.xlabel(f"Prediction: {classification_labels[predicted_class[0]]}")

    plt.subplot(1, 2, 2)
    plt.title("Perturbed Image")
    plt.imshow(triggered_image.permute(1, 2, 0).clip(0, 1))
    plt.xlabel(f"Prediction: {classification_labels[predicted_class[1]]}")

    plt.savefig(f"output/{model_name}_{dataset_name}.png")
