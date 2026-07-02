import os
import json
from collections import defaultdict

def collect_disease_and_class_ids(folder_path):
    """
    Parses JSON annotations to extract disease and pest (class) IDs.
    Handles nested and top-level annotations for data consistency checks.
    """
    disease_ids = defaultdict(list)
    class_ids = defaultdict(list)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                annotations = data.get("annotations", {})

                # Extract 'disease' ID
                disease = annotations.get("disease")
                if disease is not None:
                    disease_ids[disease].append(filename)

                # Extract 'class' (pest) ID from nested objects
                objects = annotations.get("object", [])
                for obj in objects:
                    cls = obj.get("class")
                    if cls is not None:
                        class_ids[cls].append(filename)

        except Exception as e:
            print(f"Warning: Failed to process {filename}. Error: {e}")

    return disease_ids, class_ids

def check_top_level_class_fields(folder_path):
    """
    Audits the dataset for inconsistent JSON structures where 'class'
    might be stored at the top level instead of inside the 'object' array.
    """
    suspicious_files = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            annotations = data.get("annotations", {})
            if "class" in annotations:
                # Class field found at the top level (anomaly detection)
                suspicious_files.append((filename, annotations["class"]))

        except Exception as e:
            print(f"Warning: Failed to process {filename}. Error: {e}")

    return suspicious_files

# Execution Example
if __name__ == "__main__":
    train_folder_path = r"path/to/Training/labels"
    val_folder_path = r"path/to/Validation/labels"
    
    print("Evaluating Training Set...")
    disease_map, class_map = collect_disease_and_class_ids(train_folder_path)
    
    print("\\n[Disease ID Distribution]")
    for disease_id, files in disease_map.items():
        print(f" - ID {disease_id}: {len(files)} files")

    print("\\n[Pest (Class) ID Distribution]")
    for class_id, files in class_map.items():
        print(f" - ID {class_id}: {len(files)} files")
        
    suspicious = check_top_level_class_fields(train_folder_path)
    print(f"\\nFound {len(suspicious)} files with top-level 'class' fields outside 'object' array.")