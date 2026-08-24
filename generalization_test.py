# ============================================================
# Cross-dataset generalization test — paper Section IV-E
# Zero-shot transfer: evaluate the Kaggle-trained MobileNetV2
# on the Figshare brain tumor dataset (different source, same
# 3 tumor classes -- Figshare has NO "notumor" class).
#
# BEFORE RUNNING:
# 1. Search Kaggle for "figshare brain tumor dataset" or
#    "brain tumor classification mri Cheng" -- look for a
#    version already organized into folders by class
#    (glioma / meningioma / pituitary), not raw .mat files.
# 2. Download it to your Google Drive under:
#    brain_tumor_research/figshare_dataset/
#    with subfolders however they're named (this script tries
#    to auto-detect common naming variants).
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL_PATH = '/content/drive/MyDrive/brain_tumor_research/mobilenetv2_paper_exact.keras'
FIGSHARE_DIR = '/content/drive/MyDrive/brain_tumor_research/figshare_dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Your model's original 4-class order (must match training)
FULL_CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---- Load model (compile=False -- inference only, same fix as Grad-CAM script) ----
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# ---- Auto-detect the Figshare folder structure ----
print("\nContents of figshare_dataset folder:")
if os.path.isdir(FIGSHARE_DIR):
    for item in os.listdir(FIGSHARE_DIR):
        print(" -", item)
else:
    raise FileNotFoundError(
        f"{FIGSHARE_DIR} not found. Make sure you've uploaded the Figshare "
        f"dataset to Drive at this path before running this cell."
    )

# Common naming variants across different Kaggle mirrors -- edit this
# mapping if your downloaded folder uses different names than these.
NAME_VARIANTS = {
    'glioma': ['glioma', 'glioma_tumor', 'Glioma'],
    'meningioma': ['meningioma', 'meningioma_tumor', 'Meningioma'],
    'pituitary': ['pituitary', 'pituitary_tumor', 'Pituitary'],
}

resolved_folders = {}
for canonical, variants in NAME_VARIANTS.items():
    for v in variants:
        candidate = os.path.join(FIGSHARE_DIR, v)
        if os.path.isdir(candidate):
            resolved_folders[canonical] = candidate
            break

print("\nResolved folders:", resolved_folders)
if len(resolved_folders) < 3:
    raise ValueError(
        "Could not find all 3 class folders (glioma/meningioma/pituitary). "
        "Check the actual folder names printed above and update NAME_VARIANTS."
    )

# ---- Load all Figshare images and labels ----
# Figshare classes map to indices 0=glioma, 1=meningioma, 3=pituitary
# in our model's original label space (index 2 = notumor, doesn't exist here)
LABEL_MAP = {'glioma': 0, 'meningioma': 1, 'pituitary': 3}

X_paths = []
y_true_figshare = []
for canonical, folder in resolved_folders.items():
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            X_paths.append(os.path.join(folder, fname))
            y_true_figshare.append(LABEL_MAP[canonical])

y_true_figshare = np.array(y_true_figshare)
print(f"\nTotal Figshare images found: {len(X_paths)}")
print("Class distribution:", {k: int(np.sum(y_true_figshare == v)) for k, v in LABEL_MAP.items()})

# ---- Run inference in batches ----
def load_and_preprocess(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return preprocess_input(arr)

y_pred_probs_figshare = []
for i in range(0, len(X_paths), BATCH_SIZE):
    batch_paths = X_paths[i:i+BATCH_SIZE]
    batch = np.array([load_and_preprocess(p) for p in batch_paths])
    probs = model.predict(batch, verbose=0)
    y_pred_probs_figshare.append(probs)
    if i % (BATCH_SIZE * 10) == 0:
        print(f"  Processed {i}/{len(X_paths)}...")

y_pred_probs_figshare = np.concatenate(y_pred_probs_figshare, axis=0)

# ---- Restrict predictions to the 3 classes Figshare actually has ----
# (zero out the "notumor" logit so the model can't predict a class
# that doesn't exist in this dataset, matching the paper's approach)
NOTUMOR_INDEX = 2
y_pred_probs_restricted = y_pred_probs_figshare.copy()
y_pred_probs_restricted[:, NOTUMOR_INDEX] = -np.inf
y_pred_figshare = np.argmax(y_pred_probs_restricted, axis=1)

# ---- Results ----
acc = accuracy_score(y_true_figshare, y_pred_figshare)
print(f"\n{'='*50}")
print(f"CROSS-DATASET GENERALIZATION RESULT")
print(f"Figshare zero-shot accuracy: {acc*100:.2f}%")
print(f"Paper's reported Figshare accuracy: 86.13%")
print(f"Primary dataset (Kaggle) accuracy: 94.19%")
print(f"Accuracy drop: {(0.9419 - acc)*100:.2f} percentage points")
print(f"Paper's reported drop: 8.08 percentage points")
print(f"{'='*50}")

print("\nConfusion matrix (rows=true, cols=pred, order: glioma/meningioma/notumor/pituitary):")
print(confusion_matrix(y_true_figshare, y_pred_figshare, labels=[0,1,2,3]))

# ---- Save for the audit tool ----
np.save('/content/drive/MyDrive/brain_tumor_research/y_true_figshare.npy', y_true_figshare)
np.save('/content/drive/MyDrive/brain_tumor_research/y_pred_figshare.npy', y_pred_figshare)
np.save('/content/drive/MyDrive/brain_tumor_research/y_pred_probs_figshare.npy', y_pred_probs_figshare)
print("\nSaved y_true_figshare.npy, y_pred_figshare.npy, y_pred_probs_figshare.npy to Drive.")
print("Download these and use the generalization score = this accuracy in your CRI calculation.")
