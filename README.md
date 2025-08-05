# 📁 Mise en place du projet TensorFlow Object Detection API

## 📌 1. Compiler les fichiers `.proto`

```bash
cd models/research

"C:/protoc/bin/protoc.exe" object_detection/protos/*.proto --python_out=.
```

S'assurer avant d'installer `protoc`. Regarder un tutoriel au besoin

---

## 🧠 2. Définir le chemin Python (PYTHONPATH)

```bash
set PYTHONPATH=C:\projet\models\research;C:\projet\models\research\slim;%PYTHONPATH%
```

---

## 🚀 3. Lancer l'entraînement du modèle

```bash
cd ../..

python model_main_tf2.py ^
  --pipeline_config_path=C:/projet/my_models/pipeline.config ^
  --model_dir=C:/projet/my_models/training ^
  --alsologtostderr
```

✅ **Remarques** :

- Le fichier `pipeline.config` doit être bien configuré (labels, dataset, nombre de steps, chemin des images/annotations, etc.).
- Pour monitorer l'entraînement avec TensorBoard (très intéressant) :

```bash
tensorboard --logdir=C:/projet/my_models/training
```

---

## 📤 4. Exporter le modèle entraîné

```bash
python C:/projet/models/research/object_detection/exporter_main_v2.py ^
  --input_type image_tensor ^
  --pipeline_config_path C:/projet/my_models/pipeline.config ^
  --trained_checkpoint_dir C:/projet/my_models/training ^
  --output_directory C:/projet/my_models/exported_model
```

On a maintenant un dossier `exported_model` contenant le `saved_model` prêt à l’inférence.
