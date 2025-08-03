import os
import cv2
import pytesseract
from PIL import Image
import json
from pytesseract import Output

# Spécifie le chemin vers tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\clara\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Dossiers
IMAGES_FOLDER = "C:/projet/images"
OUTPUT_FOLDER = "C:/projet/images/ocr_results"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# OCR avec sortie JSON simplifiée
def perform_ocr(image_path, output_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Impossible de lire l'image : {image_path}")
            return

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray_image)

        data = pytesseract.image_to_data(pil_image, output_type=Output.DICT)

        # Construire le JSON simplifié
        simplified_output = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if text != "" and conf > 0:
                simplified_output.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": {
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i]
                    }
                })

        # Sauvegarde
        json_output_path = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(image_path))[0] + '.json'
        )
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_output, f, ensure_ascii=False, indent=2)

        print(f"[INFO] OCR simplifié sauvegardé : {json_output_path}")

    except Exception as e:
        print(f"[ERROR] Une erreur s'est produite pour l'image {image_path}: {e}")

# Boucle sur les fichiers
for filename in os.listdir(IMAGES_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        input_path = os.path.join(IMAGES_FOLDER, filename)
        perform_ocr(input_path, OUTPUT_FOLDER)
