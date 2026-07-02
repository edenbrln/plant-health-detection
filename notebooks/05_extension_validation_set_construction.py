import os
import json
import shutil
from collections import defaultdict

# Note: As highlighted in the README's Troubleshooting section, this method of 
# augmenting the validation set by pulling directly from the training set causes 
# data leakage. It is retained here for documentation of the exploratory process.

TARGET_CLASSES = {0: 50, 12: 50, 15: 50}  # class_id: target quantity

def augment_validation_set(train_label_dir, train_img_dir, val_label_dir, val_img_dir):
    """
    Synthetically balances the validation set by copying samples from the training set.
    Warning: This introduces train/val leakage and should not be used for final evaluation.
    """
    copied_count = defaultdict(int)

    for filename in os.listdir(train_label_dir):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(train_label_dir, filename)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            objects = data.get("annotations", {}).get("object", [])
            present_classes = {obj.get("class") for obj in objects if obj.get("class") in TARGET_CLASSES}

            for cls in present_classes:
                if copied_count[cls] < TARGET_CLASSES[cls]:
                    # Copy JSON
                    shutil.copy(json_path, os.path.join(val_label_dir, filename))

                    # Copy Image
                    image_name = data.get("description", {}).get("image")
                    if image_name:
                        img_src_path = os.path.join(train_img_dir, image_name)
                        img_dst_path = os.path.join(val_img_dir, image_name)
                        if os.path.exists(img_src_path):
                            shutil.copy(img_src_path, img_dst_path)
                    
                    copied_count[cls] += 1

            # Break if all target quantities are met
            if all(copied_count[cls] >= TARGET_CLASSES[cls] for cls in TARGET_CLASSES):
                break

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\\n[Augmentation Summary]")
    for cls, count in copied_count.items():
        print(f" - Class {cls}: {count} files copied")

if __name__ == "__main__":
    # Execution Example
    train_label_dir = r"path/to/Training/labels"
    train_img_dir   = r"path/to/Training/images"
    val_label_dir   = r"path/to/Validation/labels"
    val_img_dir     = r"path/to/Validation/images"

    augment_validation_set(train_label_dir, train_img_dir, val_label_dir, val_img_dir)