import glob
import json
import numpy as np

if __name__ == "__main__":
    model_name = "model4"
    model_results = glob.glob(f"output/*/reports/{model_name}_mask_analysis.json")
    
    l1_results = {}

    for model_result_file in model_results:
        with open(model_result_file, 'r') as f:
            mask_analysis = json.load(f)

        for class_analysis in mask_analysis["class_analysis"]:
            if class_analysis["class"] not in l1_results:
                l1_results[class_analysis["class"]] = [class_analysis["l1_mask_norm"]]
            else:
                l1_results[class_analysis["class"]].append(class_analysis["l1_mask_norm"])

    l1_results_list = []
    for x in l1_results.values():
        l1_results_list.extend(x)
    l1_median_results = np.median(l1_results_list)
    l1_deviation = {x: abs(l1_results[x] - l1_median_results) for x in l1_results}

    print(l1_deviation)
