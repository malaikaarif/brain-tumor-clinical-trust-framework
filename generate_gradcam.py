# ============================================================
# Grad-CAM explainability audit — paper Section III-E / IV-C
# Generates heatmaps for the high-confidence errors found in
# the main training run, so we can visually inspect WHY the
# model was confidently wrong.
#
# Paste this as a NEW cell in the SAME Colab notebook you used
# for training (reuses the mounted Drive + dataset already there).
# If starting a fresh notebook, run the Drive-mount + dataset-
# extraction + test_generator setup from the training script first.
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

MODEL_PATH = '/content/drive/MyDrive/brain_tumor_research/mobilenetv2_paper_exact.keras'
OUTPUT_DIR = '/content/drive/MyDrive/brain_tumor_research/gradcam_samples'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---- Load the trained model ----
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ---- Recreate the test generator (same setup as training, shuffle=False
#      guarantees the same file order every time, so indices line up with
#      the already-saved y_true.npy/y_pred.npy/y_pred_probs.npy) ----
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

test_path = '/content/brain_tumor_data/Testing'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
filepaths = test_generator.filepaths
print(f"Test set: {len(filepaths)} images (should be 1600)")

# ---- Load the saved predictions (already verified, do not recompute) ----
y_true = np.load('/content/drive/MyDrive/brain_tumor_research/y_true.npy')
y_pred = np.load('/content/drive/MyDrive/brain_tumor_research/y_pred.npy')
y_pred_probs = np.load('/content/drive/MyDrive/brain_tumor_research/y_pred_probs.npy')

confidences = np.max(y_pred_probs, axis=1)
high_conf_mask = confidences >= 0.9
wrong_mask = y_pred != y_true
error_indices = np.where(high_conf_mask & wrong_mask)[0]
print(f"Found {len(error_indices)} high-confidence errors (should be 39).")

# ---- Grad-CAM implementation ----
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# Find the last conv layer name (MobileNetV2's final conv block before pooling)
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
        last_conv_layer_name = layer.name
        break
print("Using conv layer:", last_conv_layer_name)

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed)

# ---- Generate heatmaps for up to 6 high-confidence errors ----
n_samples = min(6, len(error_indices))
for i, idx in enumerate(error_indices[:n_samples]):
    img_path = filepaths[idx]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay = overlay_heatmap(img_path, heatmap)

    true_label = CLASS_NAMES[y_true[idx]]
    pred_label = CLASS_NAMES[y_pred[idx]]
    conf = confidences[idx]

    fname = f"error_{i+1}_true-{true_label}_pred-{pred_label}_conf-{conf:.2f}.png"
    overlay.save(os.path.join(OUTPUT_DIR, fname))
    print(f"Saved {fname}")

print(f"\nDone. {n_samples} Grad-CAM overlays saved to {OUTPUT_DIR}")
print("Download this folder from Drive and add it to your repo under explainability/gradcam_samples/")
