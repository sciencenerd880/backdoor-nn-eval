# Backdoor Neural Network Evaluation Project

## Overview
This project evaluates neural networks for potential backdoor attacks and helps to identify backdoor triggers. It also allows for creating backdoored models on MNIST and CIFAR-10 datasets.

## Project Structure
- `data/`: Stores the MNIST and CIFAR-10 datasets.
- `models/`: Contains the weights and architectures of the models. Original source from the google drive: https://drive.google.com/file/d/1-fI1KVbgAdRkCSLJxT7MKz-AIT7E3iDV/view
- `output/`: Contains the output for Neural Cleanse and Grad-CAM
- `src/`: Core Python scripts for loading, evaluating, and detecting backdoors.
- `semantic-backdoor-soda/`: A modified module which was originally implemented by (Bing Sun, 2024). Due to the size upload restriction (<100MB per file), the google drive link to unzip can be found here. https://drive.google.com/file/d/1STFM34q0BGak5HYnK4FuA1bvBxCHePWo/view?usp=sharing

## How to Run
1. Run `pip install -r requirements.txt` to install required packages.
2. Open Main Function in `src/main.py`, choose which model you're going to run Neural Cleanse and Grad-CAM for. By default it will run for all models.

    For Neural Cleanse, you can change the number of iteration before cutoff during neural cleanse, set as 10 by default. Change this with constant `NEURAL_CLEANSE_CUTOFF_STEP`.

    For Grad-CAM, you can change the number of visualization it will generate for each label class, set as 20 by default. Change this with constant `GRAD_CAM_NUM_VIZ_PER_CLASS`.

3. Run it using command `python src/main.py` 
    - If you want to run just the Neural Cleanse, open `src/neural_cleanse.py`, pick the model, and ruh with `python src/neural_cleanse.py`. This script by default will using iteration cutoff at 10. Change this with the constant `CUTOFF_STEP`.
    - If you want to run just Grad-CAM, open `src/grad_cam.py`, pick which CIFAR-10 models to run, and run with `python src/grad_cam.py`. This script by default will run and generate 20 numbers of images per label class. Change this with the constant `NUM_VIZ_PER_CLASS`
4. Check the output for Neural Cleanse at `output/` folder, in folders called `neural_cleanse_experiment_{time when the script was run}/` (e.g. `neural_cleanse_experiment_20241031_103155/`).
    - In these folders, you will see several subfolders that are explained in [this section](#neural-cleanse-result).
To perform SODA causality analysis and detect backdoor presence in your model, you can use the following commands. These steps will help you obtain the target class suspects based on the analysis.

5. For SODA, unzip the downloaded zipped file from https://drive.google.com/file/d/1STFM34q0BGak5HYnK4FuA1bvBxCHePWo/view?usp=sharing.

SODA: To run the causality analysis, use the following command:
```bash
cd semantic-backdoor-soda
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=CIFAR10Net --poison_type=semantic --ana_layer 3 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model2_cifar10_bd.pt --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
```

SODA: To detect backdoors based on causality analysis, use the following command:
```bash
cd semantic-backdoor-soda
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=CIFAR10Net --poison_type=semantic --confidence=3 --confidence2=0.5 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model2_cifar10_bd.pt --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
```

6. Check the output for Grad-CAM at `output/` folder, in folders called `grad_cam_{model name}/` (e.g. `grad_cam_model2/`).
    - In these folders, you will see several images that are explained in [this section](#grad-cam-result)

7. After obtaining Neural Cleanse results, run `python src/anomaly_detection_by_reference.py` to perform MAD with reference L1 backdoor detection. The results will be organized in the following structure:
    ```
    output/
    ├── neural_cleanse_experiment_model1/
    │   ├── reports/
    │   │   ├── model1_mask_analysis.json
    │   ├── triggers/
    │   └── visualizations/
    ├── neural_cleanse_experiment_model2/
    ├── neural_cleanse_experiment_model3/
    ├── neural_cleanse_experiment_model4/
    ├── neural_cleanse_experiment_model5/
    ├── neural_cleanse_experiment_reference_cifar10_1/
    │   ├── reports/
    │   │   ├── reference_cifar10_1_mask_analysis.json
    │   ├── triggers/
    │   └── visualizations/
    ├── neural_cleanse_experiment_reference_cifar10_2/
    ├── neural_cleanse_experiment_reference_cifar10_3/
    ├── neural_cleanse_experiment_reference_cifar10_4/
    └── neural_cleanse_experiment_reference_mnist_1/
     ...

    ```

## Neural Cleanse Result
Subfolders:
- reports/:
    - `experiment_log.txt`: Log of the entire experiment during the runtime of neural_cleanse.py for this specific model. 
    - `{model name}_mask_analysis.json`: Complete result of the mask analysis. For each class, there is L1 Mask Norm, Attack Success Rate, and Anomaly Index.
- triggers/:
    - all/: Contains all the triggers saved in pt file. These triggers can be loaded as a dictionary of images containing mask and delta in order to do visualization.
    - suspicious/: Contains all the suspicious trigger according to neural cleanse standards of suspicious class (ASR > 75%, Anomaly Index > 2, L1 Norm < Median throughout the class).
- visualizations/:
    - `NeuralCleanse_MaskL1Reg_Model_{model name}_Target_{target class}.png`: Image of the generated trigger mask in grayscale.
    - `NeuralCleanse_MaskL1Reg_Model_{model name}_Target_{target class}_rgb.png`: Image of the generated trigger mask in RGB.

Files:
- `experiment_params.json`: Stores the parameters that's used in the run which includes model_name, dataset_name, timestamp, learning rate, number of steps, lambda l1, and attack success rate (ASR) threshold.

## Grad-CAM Result
- `image_{label class}_{data index}.png`: Image containing the original image, label and prediction, and also the Grad-CAM result image that's generated from the last layer.

## Additional Notes
1. Grad-CAM can only run for CIFAR-10 models because they are Convolutional Neural Networks (CNN). Grad-CAM relies on the Convolutional layers, which it will then leverages spatial information present in those convolutional layers to create visual explanations.
2. Because of the timestamp, your neural cleanse results will not get overwritten
3. Grad-CAM results WILL get overwritten, however, because the folder naming only uses model name.

## References
Bing Sun, J. S. (2024). Neural Network Semantic Backdoor Detection and Mitigation: A Causality-Based Approach. 33rd USENIX Security Symposium (USENIX Security 24), 2883-2900. Retrieved from https://www.usenix.org/conference/usenixsecurity24/presentation/sun-bing

<!-- ## Requirements
Given a (third-party trained) neural network, your task is to evaluate whether there are backdoors embedded in the neural network. 
We will provide multiple backdoored (or not) neural networks trained on the MNIST, CIFAR-10, and CIFAR-100 datasets. 
You as a team will provide us one backdoored neural network trained on the same datasets.
You will be evaluated in terms of (1) whether an alarm is triggered if there is a backdoor; and (2) whether the backdoor trigger is successfully identified.  

- Each folder contains a backdoor model trained with different backdoor attacks, triggers, and targets.
- Model 1 is trained on MNIST dataset. The rest are trained on CIFAR10 dataset.
- The architectures of the models can be found in the according python files (model_mnist.py for the MNIST model and model_cifar10.py for the CIFAR10 models). -->
