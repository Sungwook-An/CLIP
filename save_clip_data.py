import os
import csv

# 최상위 경로 설정
base_path = '/database/PCB_DATASET/crop_images'
output_csv = 'clip_data.csv'

# CSV 파일 작성
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'text', 'label'])  # CSV 헤더 작성

    # 각 클래스 폴더 탐색
    for idx, class_folder in enumerate(os.listdir(base_path)):
        class_path = os.path.join(base_path, class_folder)
        
        if os.path.isdir(class_path):
            # 각 파일 탐색
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                
                if os.path.isfile(image_path):
                    # 이미지 파일 경로와 텍스트 설명 작성
                    text_description = f'A photo of class {idx}'
                    writer.writerow([image_path, text_description, idx])

print(f"CSV 파일이 '{output_csv}'에 성공적으로 저장되었습니다.")