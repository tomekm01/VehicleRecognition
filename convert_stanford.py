import scipy.io
import pandas as pd
import os

def convert_stanford_cars_to_csv():
    print("Loading .mat files...")
    meta = scipy.io.loadmat(r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\car_devkit\devkit\SC_cars_meta.mat")
    mat_test = scipy.io.loadmat(r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\car_devkit\devkit\SC_cars_test_annos.mat")
    mat_train = scipy.io.loadmat(r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\car_devkit\devkit\SC_cars_train_annos.mat")

    labels = [c[0] for c in meta['class_names'][0]]

    train_data = []
    
    for anno in mat_train['annotations'][0]:
        fname = str(anno['fname'][0]).replace('\\', '/').split('/')[-1]
        
        class_id = int(anno['class'][0][0])
        class_name = labels[class_id - 1]
        
        train_data.append({
            'filename': fname,
            'bbox_x1': int(anno['bbox_x1'][0][0]),
            'bbox_y1': int(anno['bbox_y1'][0][0]),
            'bbox_x2': int(anno['bbox_x2'][0][0]),
            'bbox_y2': int(anno['bbox_y2'][0][0]),
            'class_id': class_id,
            'class_name': class_name
        })

    df_train = pd.DataFrame(train_data)
    df_train.to_csv('cars_train.csv', index=False)

    test_data = []
    
    for anno in mat_test['annotations'][0]:
        fname = str(anno['fname'][0]).replace('\\', '/').split('/')[-1]
        
        test_data.append({
            'filename': fname,
            'bbox_x1': int(anno['bbox_x1'][0][0]),
            'bbox_y1': int(anno['bbox_y1'][0][0]),
            'bbox_x2': int(anno['bbox_x2'][0][0]),
            'bbox_y2': int(anno['bbox_y2'][0][0])
        })

    df_test = pd.DataFrame(test_data)
    df_test.to_csv('cars_test.csv', index=False)

if __name__ == "__main__":
    convert_stanford_cars_to_csv()