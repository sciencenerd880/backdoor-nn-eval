import json
import os
import numpy as np
import datetime

def load_json_report(filepath):
    """load analysis report from json file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_class_l1_norm(report_data, class_name):
    """extract l1_mask_norm for a specific class from report data"""
    for class_analysis in report_data['class_analysis']:
        if class_analysis['class'] == class_name:
            return class_analysis['l1_mask_norm']
    return None

def get_reference_l1_norms(reference_reports, class_name, target_l1_norm):
    """collect l1_mask_norms for a specific class from all reference models and target model"""
    # include target model's l1_norm in the reference values
    reference_values = [get_class_l1_norm(report, class_name) for report in reference_reports]
    reference_values.append(target_l1_norm)
    return reference_values

def compute_mad_stats(values):
    """compute median and mad for a list of values"""
    consistency_constant = 1.4826  # constant for MAD calculation
    median = np.median(values)
    mad = consistency_constant * np.median(np.abs(np.array(values) - median))
    return median, mad

def compute_anomaly_index(value, median, mad):
    """compute anomaly index for a value"""
    return np.abs(value - median) / mad if mad != 0 else 0

def detect_anomalies_by_reference(model_name, model_report, reference_reports, output_dir):
    """perform anomaly detection using reference models"""
    results = {
        'model_name': model_name,
        'analysis_timestamp': str(datetime.datetime.now()),
        'class_analysis': [],
        'suspicious_classes': []
    }
    
    # analyze each class
    for class_analysis in model_report['class_analysis']:
        class_name = class_analysis['class']
        target_l1_norm = class_analysis['l1_mask_norm']
        
        # get reference values including target model's value
        reference_l1_norms = get_reference_l1_norms(reference_reports, class_name, target_l1_norm)
        
        # compute statistics
        median_norm, mad = compute_mad_stats(reference_l1_norms)
        anomaly_index = compute_anomaly_index(target_l1_norm, median_norm, mad)
        
        # store analysis results
        class_result = {
            'class': class_name,
            'l1_mask_norm': target_l1_norm,
            'reference_median': median_norm,
            'reference_mad': mad,
            'anomaly_index': anomaly_index,
            'attack_success_rate': class_analysis['attack_success_rate']
        }
        results['class_analysis'].append(class_result)
        
        # check if suspicious
        if (anomaly_index > 2 and 
            target_l1_norm < median_norm and 
            class_analysis['attack_success_rate'] > 75):
            results['suspicious_classes'].append(class_result)
    
    # save results
    output_path = os.path.join(output_dir, f"{model_name}_reference_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    # create output directory
    output_dir = "output/anomaly_detection_by_reference/reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # define model groups with their reference models
    model_groups = {
        'mnist': {
            'models': [
                ("model1", "output/neural_cleanse_experiment_model1/reports/model1_mask_analysis.json")
            ],
            'references': [
                "output/neural_cleanse_experiment_reference_mnist_1/reports/reference_mnist_1_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_mnist_2/reports/reference_mnist_2_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_mnist_3/reports/reference_mnist_3_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_mnist_4/reports/reference_mnist_4_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_mnist_5/reports/reference_mnist_5_mask_analysis.json"
            ]
        },
        'cifar10': {
            'models': [
                ("model2", "output/neural_cleanse_experiment_model2/reports/model2_mask_analysis.json"),
                ("model3", "output/neural_cleanse_experiment_model3/reports/model3_mask_analysis.json"),
                ("model4", "output/neural_cleanse_experiment_model4/reports/model4_mask_analysis.json"),
                ("model5", "output/neural_cleanse_experiment_model5/reports/model5_mask_analysis.json")
            ],
            'references': [
                "output/neural_cleanse_experiment_reference_cifar10_1/reports/reference_cifar10_1_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_cifar10_2/reports/reference_cifar10_2_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_cifar10_3/reports/reference_cifar10_3_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_cifar10_4/reports/reference_cifar10_4_mask_analysis.json",
                "output/neural_cleanse_experiment_reference_cifar10_5/reports/reference_cifar10_5_mask_analysis.json"
            ]
        }
    }
    
    # process each model group
    for dataset, group in model_groups.items():
        print(f"\nProcessing {dataset.upper()} models...")
        
        # load reference reports for this dataset
        reference_reports = [load_json_report(path) for path in group['references']]
        
        # analyze each model in the group
        for model_name, model_path in group['models']:
            print(f"\nAnalyzing {model_name}...")
            try:
                model_report = load_json_report(model_path)
                results = detect_anomalies_by_reference(
                    model_name,
                    model_report,
                    reference_reports,
                    output_dir
                )
                
                # print summary
                print(f"Analysis completed for {model_name}")
                if results['suspicious_classes']:
                    print("Suspicious classes detected:")
                    for suspicious in results['suspicious_classes']:
                        print(f"  Class {suspicious['class']}:")
                        print(f"    L1 Norm: {suspicious['l1_mask_norm']:.4f}")
                        print(f"    Reference Median: {suspicious['reference_median']:.4f}")
                        print(f"    Anomaly Index: {suspicious['anomaly_index']:.4f}")
                        print(f"    ASR: {suspicious['attack_success_rate']:.2f}%")
                else:
                    print("No suspicious classes detected")
                    
            except Exception as e:
                print(f"Error analyzing {model_name}: {str(e)}")

if __name__ == "__main__":
    main()
