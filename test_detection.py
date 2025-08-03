import os
import json
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# === CONFIGURATION ===
MODEL_PATH = "C:/projet/my_models/exported_model/saved_model"
LABEL_MAP_PATH = "C:/projet/annotations/label_map.pbtxt"
IMAGES_FOLDER = "C:/projet/images"
OUTPUT_FOLDER = "C:/projet/images/results_images"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === CHARGEMENT DU MODÈLE ===
detect_fn = tf.saved_model.load(MODEL_PATH)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# === FONCTION DE TRAITEMENT D'UNE IMAGE ===
def process_image(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    # Visualisation sur l'image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.3,
        agnostic_mode=False
    )

    # Sauvegarde de l'image annotée
    Image.fromarray(image_np).save(output_path)
    print(f"[INFO] Image sauvegardée : {output_path}")

    # Préparer les données à sauvegarder en JSON
    detection_results = []
    for box, cls, score in zip(boxes, classes, scores):
        if score >= 0.3:
            detection_results.append({
                "box": box.tolist(),  # ymin, xmin, ymax, xmax (normalisé 0-1)
                "class_id": int(cls),
                "class_name": category_index[cls]['name'] if cls in category_index else "N/A",
                "score": float(score)
            })

    return detection_results

# === BOUCLE SUR LES IMAGES DU DOSSIER ===
for filename in os.listdir(IMAGES_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(IMAGES_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        results = process_image(input_path, output_path)

        # Sauvegarder les résultats dans un fichier JSON
        json_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + ".json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Résultats JSON sauvegardés : {json_path}")
