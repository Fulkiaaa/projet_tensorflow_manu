import os
import cv2
import numpy as np
import tensorflow as tf
import keras_ocr
import json

# Chemins
PATH_TO_MODEL = "my_models/exported_model/saved_model"
PATH_TO_IMAGES = "images"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionnaire des classes
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

# Chargement du modèle
print("Chargement du modèle de détection...")
try:
    detect_fn = tf.saved_model.load(PATH_TO_MODEL)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    exit()

# Chargement pipeline Keras-OCR
print("Chargement du pipeline OCR...")
try:
    pipeline = keras_ocr.pipeline.Pipeline()
    print("Pipeline OCR prêt.")
except Exception as e:
    print(f"Erreur lors du chargement du pipeline OCR: {e}")
    exit()

# Prédiction sur chaque image
results = []
for filename in os.listdir(PATH_TO_IMAGES):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(PATH_TO_IMAGES, filename)
    print(f"\nTraitement de l'image: {image_path}")

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur: Impossible de lire l'image {image_path}")
            continue

        input_rgb = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        input_tensor = tf.convert_to_tensor(input_rgb[None, ...], dtype=tf.uint8)

        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        h, w, _ = image.shape
        image_result = {'image': filename, 'fields': []}

        for i in range(num_detections):
            score = detections['detection_scores'][i]
            if score < 0.05:
                continue

            class_id = detections['detection_classes'][i]
            label = label_map.get(class_id, "unknown")
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
            left, right = int(xmin * w), int(xmax * w)
            top, bottom = int(ymin * h), int(ymax * h)

            # Clamp les coordonnées
            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            if left >= right or top >= bottom:
                continue

            # Crop & OCR
            cropped = image[top:bottom, left:right]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            try:
                ocr_result = pipeline.recognize([cropped_rgb])[0]
                text = " ".join([word[0] for word in ocr_result])
            except Exception as e:
                print(f"OCR échoué: {e}")
                text = ""

            image_result['fields'].append({
                'label': label,
                'class_id': int(class_id),
                'score': float(score),
                'box': [left, top, right, bottom],
                'text': text
            })

            # Dessin (facultatif)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {text}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sauvegarde image annotée
        output_img = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_img, image)
        results.append(image_result)

    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_path}: {e}")

# Sauvegarde JSON
with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nFini. Résultats sauvegardés dans: {OUTPUT_DIR}")
