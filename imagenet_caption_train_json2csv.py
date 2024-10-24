import json
import csv

with open('/root/LGD2024/examples_old/imagenet/imagenet_caption_train.json', 'r') as json_file:
    data = json.load(json_file)

csv_columns = ['image_id', 'caption_enriched']

for item in data:
    item['image_id'] = item['image_id'] + '.JPEG'

csv_file = "/root/LGD2024/examples_old/imagenet/imagenet_caption_train.csv"
try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(data)
    print(f"CSV file '{csv_file}' created successfully with '.JPEG' appended to image_id.")
except IOError:
    print("I/O error while creating CSV file.")