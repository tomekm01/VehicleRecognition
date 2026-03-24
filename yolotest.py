import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

def extract_and_display_car(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img, verbose=False) 

    car_boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) in [2, 5, 7]: # cars, buses, trucks
                car_boxes.append(box.xyxy[0].cpu().numpy())

    if not car_boxes:
        print("No car detected in this image.")
        return

    def calculate_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
    largest_car_box = max(car_boxes, key=calculate_area)
    x1, y1, x2, y2 = map(int, largest_car_box)

    cropped_img = img_rgb[y1:y2, x1:x2]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    rect_img = img_rgb.copy()
    cv2.rectangle(rect_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    plt.imshow(rect_img)
    plt.title('Original Image (YOLO Detection)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_img)
    plt.title('Cropped Car Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


sample_image_path = "./compcars_1.jpg"
extract_and_display_car(sample_image_path)
sample_image_path = "./compcars_2.jpg"
extract_and_display_car(sample_image_path)
sample_image_path = "./stanford_1.jpg"
extract_and_display_car(sample_image_path)
sample_image_path = "./stanford_2.jpg"
extract_and_display_car(sample_image_path)