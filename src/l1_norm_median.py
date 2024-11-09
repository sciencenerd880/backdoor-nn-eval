import glob
import json

if __name__ == "__main__":
    model_name = "model2"
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

    print(l1_results)
