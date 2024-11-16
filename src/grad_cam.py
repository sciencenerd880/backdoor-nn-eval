import matplotlib.pyplot as plt
import os
from load_model import load_model
from load_dataset import load_dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def grad_cam_viz(model_name, dataset_name, num_viz_per_class=20):
    model, device = load_model(model_name, dataset_name)

    if dataset_name == "cifar10":
        classification_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classification_labels = [str(i) for i in range(10)]

    # Assume we use the final conv layer
    target_layers = [model.conv6]

    # Sample image from CIFAR-10 dataset
    cifar10 = load_dataset(dataset_name, train=False)

    # Initialize GradCAM with the model and target layer
    with GradCAM(model=model, target_layers=target_layers) as grad_cam:
        # pick 5 data for each class
        data_to_try = {}
        for idx, (image, label) in enumerate(cifar10):
            if label not in data_to_try:
                data_to_try[label] = [(idx, image, label)]
            elif len(data_to_try[label]) < num_viz_per_class:
                data_to_try[label].append((idx, image, label))

        for every_label in data_to_try:
            for idx, img, label in data_to_try[every_label]:
                input_tensor = img.unsqueeze(0)  # Add batch dimension

                # Generate heatmap for the predicted class
                model.eval()
                output = model(input_tensor)
                predicted_class = output.argmax(dim=1).item()
                targets = [ClassifierOutputTarget(predicted_class)]

                heatmap = grad_cam(input_tensor=input_tensor, targets=targets)
                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(img.permute(1, 2, 0))
                plt.xlabel(f"Label: {classification_labels[label]}\nPredicted: {classification_labels[predicted_class]}")
                
                plt.subplot(1, 2, 2)
                plt.title("Grad-CAM")
                plt.imshow(img.permute(1, 2, 0))
                plt.imshow(heatmap.squeeze(), cmap="jet", alpha=0.5)

                if not os.path.exists(f"output/grad_cam_{model_name}/"):
                    os.mkdir(f"output/grad_cam_{model_name}/")
                plt.savefig(f"output/grad_cam_{model_name}/image_{classification_labels[label]}_{idx}.png")
                plt.close()


if __name__ == "__main__":
    models = [
        # ("reference_cifar10_2", "cifar10"),
        # ("reference_cifar10_3", "cifar10"),
        # ("reference_cifar10_4", "cifar10"),
        # ("reference_cifar10_5", "cifar10"),
        ("model2", "cifar10"),
        # ("model3", "cifar10"),
        # ("model4", "cifar10"),
        # ("model5", "cifar10"),
    ]

    NUM_VIZ_PER_CLASS = 20

    for model_name, dataset_name in models:
        grad_cam_viz(model_name, dataset_name, NUM_VIZ_PER_CLASS)
