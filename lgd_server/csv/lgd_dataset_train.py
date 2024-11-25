import os
import csv

base_dir = "/data001/dlpusers/kangx80/PROJECT/AI_SHARE/SOGANG-LGD_SHARE/VFM_data/train"
output_csv = "lgd_dataset_train.csv"

class_folders = ["class_1", "class_2", "class_3"]
class_descriptions = {
    "class_1": "first class",
    "class_2": "second class",
    "class_3": "third class"
}
labels = {class_name: idx + 1 for idx, class_name in enumerate(class_folders)}

with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image", "text", "label"]) 
    
    for class_name, label in labels.items():
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            print(class_path)
            continue
        
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(".jpg"):
                image_path = os.path.join(class_path, file_name)
                text_description = f"a picture of {class_descriptions[class_name]}"
                writer.writerow([image_path, text_description, label])

print(f"CSV file generated: {output_csv}")
