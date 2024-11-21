import os
import sys
import csv
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)
from imagenet_class import CLASSNAMES, TEMPLATES

# 최상위 경로 설정
base_path = '/database/Data/CLS-LOC/train/'
output_csv = 'imagenet_picture_train.csv'

def extract_number(item):
    return int(item[1:])

# CSV 파일 작성
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'text', 'label'])  # CSV 헤더 작성

    dir_list = os.listdir(base_path)
    sorted_list = sorted(dir_list, key=extract_number)

    # 각 클래스 폴더 탐색
    for idx, class_folder in enumerate(sorted_list):
        class_path = os.path.join(base_path, class_folder)
        
        if os.path.isdir(class_path):
            # 각 파일 탐색
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                
                if os.path.isfile(image_path):
                    # 이미지 파일 경로와 텍스트 설명 작성
                    for t in TEMPLATES:
                        text_description = t.format(CLASSNAMES[idx])
                        writer.writerow([image_path, text_description, idx])
        else:
            print(f"경고: {class_folder} 폴더가 존재하지 않습니다.")

print(f"CSV 파일이 '{output_csv}'에 성공적으로 저장되었습니다.")
