import pandas as pd
import scipy.io
from pathlib import Path

def build_compcars_csv():
    df_attributes = pd.read_csv(r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\misc\attributes.txt", sep=r'\s+')
    
    mat_names = scipy.io.loadmat(r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\misc\make_model_name.mat")
    make_names = [m[0][0] if len(m) > 0 and len(m[0]) > 0 else "" for m in mat_names['make_names']]
    model_names = [m[0][0] if len(m) > 0 and len(m[0]) > 0 else "" for m in mat_names['model_names']]
    
    label_dir = Path(r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\label")
    data = []

    for txt_path in label_dir.rglob('*.txt'):
        parts = txt_path.parts
        
        if len(parts) < 5:
            continue
            
        make_id = int(parts[-4])
        model_id = int(parts[-3])
        year = parts[-2]
        
        img_name = parts[-1].replace('.txt', '.jpg')
        relative_img_path = f"{make_id}/{model_id}/{year}/{img_name}"

        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()

        if len(lines) >= 3:
            viewpoint = int(lines[0])
            bbox = lines[2].split()
            
            make_name = make_names[make_id - 1] if 0 <= make_id - 1 < len(make_names) else str(make_id)
            model_name = model_names[model_id - 1] if 0 <= model_id - 1 < len(model_names) else str(model_id)
            class_name = f"{make_name} {model_name} {year}"

            data.append({
                'filename': relative_img_path,
                'class_name': class_name.strip(),
                'bbox_x1': int(bbox[0]),
                'bbox_y1': int(bbox[1]),
                'bbox_x2': int(bbox[2]),
                'bbox_y2': int(bbox[3]),
                'make_id': make_id,
                'model_id': model_id,
                'viewpoint': viewpoint
            })

    df_labels = pd.DataFrame(data)
    df_final = pd.merge(df_labels, df_attributes, on='model_id', how='left')
    df_final.to_csv('compcars_dataset.csv', index=False)

if __name__ == '__main__':
    build_compcars_csv()