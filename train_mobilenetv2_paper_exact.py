# ============================================================
# MobileNetV2 training — matches Beyond Accuracy paper exactly
# Section III-A (preprocessing/augmentation), III-B (Focal Loss),
# III-C (architecture), III-D (training protocol), Table IV.
# Paste this as a single cell in Colab. Needs the same
# archive.zip / brain_tumor_research/ Drive setup Iqra used.
# ============================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import zipfile, os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support

# ---- Reproducibility: paper uses seed 42 ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- Extract dataset (same path Iqra used) ----
zip_path = '/content/drive/MyDrive/brain_tumor_research/archive.zip'
extract_path = '/content/brain_tumor_data'
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
print("Dataset ready.")

train_path = '/content/brain_tumor_data/Training'
test_path = '/content/brain_tumor_data/Testing'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Table IV

# ---- Preprocessing: paper says "normalized with ImageNet statistics" ----
# preprocess_input for MobileNetV2 does the standard ImageNet-compatible
# scaling this family of models expects (not plain 1/255 rescale).
# ---- Augmentation: paper Section III-A, verbatim ----
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,          # ±20°
    width_shift_range=0.15,     # shift ±15%
    height_shift_range=0.15,
    zoom_range=0.15,            # zoom ±15%
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2        # 20% validation split (paper: "reserved for early stopping")
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', seed=SEED
)
val_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', seed=SEED
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
print(f"Train: {train_generator.samples}, Val: {val_generator.samples}, Test: {test_generator.samples}")
CLASS_NAMES = list(train_generator.class_indices.keys())
print("Classes:", CLASS_NAMES)

# ---- Focal Loss: paper Eq. (1), gamma=2.0, alpha=0.25 ----
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss_fn

# ---- Architecture: paper Section III-C ----
# "final 50 layers of each trunk were unfrozen for fine-tuning;
#  all deeper layers were frozen"
# Head: GlobalAvgPool -> BatchNorm -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(4, Softmax)
def build_paper_model(num_classes=4):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(base.input, out)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),   # Table IV: initial LR 1e-4
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model

# ---- Callbacks: Table IV ----
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    ModelCheckpoint(
        '/content/drive/MyDrive/brain_tumor_research/mobilenetv2_paper_exact.keras',
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

print("\nTraining MobileNetV2 (paper-matched protocol)...")
model = build_paper_model(len(CLASS_NAMES))
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,  # Table IV max epochs, early stopping will likely cut this short
    callbacks=callbacks,
    verbose=1
)

# ---- Evaluate on the held-out 1,600-image test set ----
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
acc = np.mean(y_pred == y_true)

print(f"\n{'='*50}")
print(f"RESULT (compare against paper Table V/VI: Acc 94.69%, ECE 0.0479)")
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1 Score:  {f1*100:.2f}%")
print(f"{'='*50}")

# ---- Save arrays for MedTrust-Audit ----
np.save('/content/drive/MyDrive/brain_tumor_research/y_true.npy', y_true)
np.save('/content/drive/MyDrive/brain_tumor_research/y_pred.npy', y_pred)
np.save('/content/drive/MyDrive/brain_tumor_research/y_pred_probs.npy', y_pred_probs)
model.save('/content/drive/MyDrive/brain_tumor_research/mobilenetv2_paper_exact.keras')
print("Saved y_true.npy, y_pred.npy, y_pred_probs.npy, and the model to Drive.")
print("Download these 3 .npy files and drop them straight into MedTrust-Audit.")
