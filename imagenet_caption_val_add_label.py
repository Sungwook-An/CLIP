import os
import pandas as pd

csv_file_path = '/root/LGD2024/examples_old/imagenet/imagenet_caption_val.csv'

image_base_path = '/database/Data/CLS-LOC/val/'

folders = sorted([d for d in os.listdir(image_base_path) if os.path.isdir(os.path.join(image_base_path, d))])
folder_to_label = {folder: idx for idx, folder in enumerate(folders)}

df = pd.read_csv(csv_file_path)

def update_image_id(image_id):
    image_file = image_id.split('/')[-1]
    
    for folder in folders:
        folder_path = os.path.join(image_base_path, folder)
        image_path = os.path.join(folder_path, image_file)
        if os.path.exists(image_path):
            return image_path
    return None

df['image_id'] = df['image_id'].apply(update_image_id)

def get_label_from_image_id(image_id):
    if image_id is not None:
        folder_name = image_id.split('/')[-2]
        return folder_to_label.get(folder_name)
    return None

df['label'] = df['image_id'].apply(get_label_from_image_id)

df.rename(columns={'image_id': 'image', 'caption_enriched': 'text'}, inplace=True)

df.to_csv('/root/LGD2024/examples_old/imagenet/imagenet_caption_val_with_labels.csv', index=False)

print("csv file saved.")