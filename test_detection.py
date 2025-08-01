import os
import cv2
import numpy as np
import tensorflow as tf
import keras_ocr
import json
import csv

# Chemins
PATH_TO_SAVED_MODEL = "my_models/exported_model/saved_model"
PATH_TO_IMAGES = "images"
PATH_TO_OUTPUT = "images_output"

os.makedirs(PATH_TO_OUTPUT, exist_ok=True)

# Charger modèle TensorFlow (détection)
print("Loading detection model...")
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print("Model loaded!")

# Charger pipeline Keras-OCR (reconnaissance texte)
print("Loading Keras-OCR pipeline...")
pipeline = keras_ocr.pipeline.Pipeline()
print("Keras-OCR pipeline loaded!")

# Label map id -> label
label_map = {
    1: "date_of_birth",
    2: "date_of_expiration",
    3: "document_number",
    4: "given_names",
    5: "nationality",
    6: "place_of_birth",
    7: "sex",
    8: "surname"
}

# Lister images
image_paths = [os.path.join(PATH_TO_IMAGES, f) for f in os.listdir(PATH_TO_IMAGES)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

results = []

for image_path in image_paths:
    print(f"Processing {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        continue

    input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(type(input_rgb), input_rgb.shape) 

    input_rgb = np.array(input_rgb)
    input_tensor = tf.convert_to_tensor(input_rgb, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Prédiction détection
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    height, width, _ = image.shape

    image_result = {
        'image_path': image_path,
        'fields': []
    }

    for i in range(num_detections):
        score = detections['detection_scores'][i]
        cls = int(detections['detection_classes'][i])
        label = label_map.get(cls, "unknown")
        box = detections['detection_boxes'][i]

        print(f"Detection {i}: class={label}, score={score:.3f}, box={box}")
        if score < 0.5: # Seuil de confiance
            continue

        cls = int(detections['detection_classes'][i])
        label = label_map.get(cls, "unknown")

        box = detections['detection_boxes'][i]
        ymin, xmin, ymax, xmax = box
        left, right = int(xmin * width), int(xmax * width)
        top, bottom = int(ymin * height), int(ymax * height)

        # Crop zone détectée
        cropped = image[top:bottom, left:right]

        # OCR sur la zone détectée avec Keras-OCR
        try:
            ocr_result = pipeline.recognize([cropped])[0]  # Liste avec un élément
            # ocr_result est une liste de tuples : ((x0,y0),(x1,y1),...), texte reconnu

            # Concatène tous les textes détectés dans la zone
            text = " ".join([word[0] for word in ocr_result]) if ocr_result else ""
        except Exception as e:
            print(f"OCR failed for {image_path} box {i}: {e}")
            text = ""

        image_result['fields'].append({
            'class_id': cls,
            'label': label,
            'score': float(score),
            'box': [left, top, right, bottom],
            'text': text
        })

        # Optionnel : dessiner sur image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({score:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sauvegarder image annotée
    output_img_path = os.path.join(PATH_TO_OUTPUT, os.path.basename(image_path))
    cv2.imwrite(output_img_path, image)

    results.append(image_result)

# Enregistrer résultats en JSON
with open(os.path.join(PATH_TO_OUTPUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Enregistrer résultats en CSV (une ligne par champ détecté)
with open(os.path.join(PATH_TO_OUTPUT, "results.csv"), "w", newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image_path', 'class_id', 'label', 'score', 'box', 'text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for img_res in results:
        for field in img_res['fields']:
            writer.writerow({
                'image_path': img_res['image_path'],
                'class_id': field['class_id'],
                'label': field['label'],
                'score': field['score'],
                'box': field['box'],
                'text': field['text']
            })

print("Processing done, results saved to", PATH_TO_OUTPUT)
