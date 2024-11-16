from neural_cleanse import neural_cleanse
from grad_cam import grad_cam_viz


if __name__ == "__main__":
    models = [
        ("reference_mnist_1", "mnist"),
        ("reference_mnist_2", "mnist"),
        ("reference_mnist_3", "mnist"),
        ("reference_mnist_4", "mnist"),
        ("reference_mnist_5", "mnist"),
        ("reference_cifar10_2", "cifar10"),
        ("reference_cifar10_3", "cifar10"),
        ("reference_cifar10_4", "cifar10"),
        ("reference_cifar10_5", "cifar10"),
        ("model1", "mnist"),
        ("model2", "cifar10"),
        ("model3", "cifar10"),
        ("model4", "cifar10"),
        ("model5", "cifar10"),
    ]
    
    # Run Neural Cleanse for All Models
    for model_name, dataset_name in models:
        neural_cleanse(model_name, dataset_name)
        # TODO: Integrate SODA
        grad_cam_viz(model_name, dataset_name)
