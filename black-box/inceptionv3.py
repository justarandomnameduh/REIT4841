import pandas as pd
import numpy as np
import os
import shutil # For potentially creating structured directories if needed, though flow_from_dataframe avoids this
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
IMG_HEIGHT, IMG_WIDTH = 299, 299  # InceptionV3 default input size
BATCH_SIZE = 32
EPOCHS = 30 # Adjust as needed, start with a moderate number
LEARNING_RATE = 0.0001
IMAGE_DIR = '../ham-concept/ham_concept_dataset/ISIC2018_Task3_Training_Input'
TRAIN_CSV_PATH = '../ham-concept/ham_concept_dataset/Datasets/metadata/train.csv'
VAL_CSV_PATH = '../ham-concept/ham_concept_dataset/Datasets/metadata/val.csv'
MODEL_WEIGHTS_PATH = 'inceptionv3_isic2018_best.weights.h5' # Keras 3 uses .weights.h5


# --- 1. Data Preparation ---
def load_and_prepare_df(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    
    df_processed = df[['image_id', 'benign_malignant']].copy()
    
    df_processed['filename'] = df_processed['image_id'].apply(lambda x: x + '.jpg')
    
    # Ensure the target column 'benign_malignant' is string type for ImageDataGenerator
    # It should contain '0' for nevus and '1' for melanoma as per the problem description.
    df_processed['benign_malignant'] = df_processed['benign_malignant'].astype(int)
    
    print(f"Loaded {csv_path}:")
    print(f"  Total rows: {len(df_processed)}")
    if not df_processed.empty:
        print(f"  Melanoma (1): {len(df_processed[df_processed['benign_malignant'] == 1])}")
        print(f"  Nevus (0): {len(df_processed[df_processed['benign_malignant'] == 0])}")
    
    return df_processed[['filename', 'benign_malignant']]

train_df = load_and_prepare_df(TRAIN_CSV_PATH, IMAGE_DIR)
val_df = load_and_prepare_df(VAL_CSV_PATH, IMAGE_DIR)

# --- 2. Data Augmentation and Generators ---
# For training data, apply augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=inception_v3_preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation data, only apply preprocessing
val_datagen = ImageDataGenerator(
    preprocessing_function=inception_v3_preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_DIR,
    x_col='filename',
    y_col='benign_malignant',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary', # Since we have 0 and 1
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=IMAGE_DIR,
    x_col='filename',
    y_col='benign_malignant',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Important for evaluation
)

# Verify class indices
print("Class indices from train_generator:", train_generator.class_indices)
# Expected: {'0': 0, '1': 1} where 1 is melanoma

# --- 3. Model Building (InceptionV3) ---
# Load pre-trained InceptionV3 model without the top classification layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x) # Regularization
predictions = Dense(1, activation='sigmoid')(x) # Binary classification: 1 unit, sigmoid activation

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile the Model ---
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

# --- 5. Callbacks ---
checkpoint = ModelCheckpoint(MODEL_WEIGHTS_PATH,
                             monitor='val_auc', # Monitor validation AUC
                             save_best_only=True,
                             save_weights_only=True, # Save only weights
                             mode='max', # For AUC, higher is better
                             verbose=1)

early_stopping = EarlyStopping(monitor='val_auc',
                               patience=10, # Number of epochs with no improvement
                               restore_best_weights=True, # Restore model weights from the epoch with the best value
                               mode='max',
                               verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_auc',
                              factor=0.2, # Factor by which learning rate will be reduced
                              patience=5,
                              min_lr=1e-7, # Lower bound on the learning rate
                              mode='max',
                              verbose=1)

callbacks_list = [checkpoint, early_stopping, reduce_lr]

# --- 6. Training ---
# Calculate class weights for imbalanced dataset (optional but often helpful)
# Class 0 (Nevus) vs Class 1 (Melanoma)
# If class_indices are {'0':0, '1':1}
# count_0 = np.sum(train_generator.classes == 0)
# count_1 = np.sum(train_generator.classes == 1)
# total = count_0 + count_1
# weight_for_0 = (1 / count_0) * (total / 2.0)
# weight_for_1 = (1 / count_1) * (total / 2.0)
# class_weight = {0: weight_for_0, 1: weight_for_1}
# print(f"Class weights: {class_weight}")
# For simplicity, we'll proceed without class weights first. If results are poor for minority class, add this.

print("\nStarting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks_list
    # class_weight=class_weight # Uncomment if using class weights
)

# --- 7. Evaluation ---
print("\nLoading best weights for evaluation...")
model.load_weights(MODEL_WEIGHTS_PATH) # Keras 3 automatically restores best weights if EarlyStopping's restore_best_weights=True

print("\nEvaluating on validation set:")
eval_results = model.evaluate(val_generator, steps=val_generator.samples // BATCH_SIZE)
print(f"Validation Loss: {eval_results[0]:.4f}")
print(f"Validation Accuracy: {eval_results[1]:.4f}")
print(f"Validation AUC: {eval_results[2]:.4f}")


# Get predictions
y_pred_proba = model.predict(val_generator, steps=val_generator.samples // BATCH_SIZE +1) # +1 to ensure all samples are predicted
# The generator might not yield exactly val_generator.samples if it's not perfectly divisible by BATCH_SIZE
# So we take predictions up to the number of samples
y_pred_proba = y_pred_proba[:val_generator.samples]


y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()
y_true = val_generator.classes # True labels

# Ensure y_true is also sliced if y_pred_proba was sliced
y_true = y_true[:len(y_pred_classes)]


# Classification Report
print("\nClassification Report:")
# target_names should correspond to class_indices: 0 for Nevus, 1 for Melanoma
# train_generator.class_indices gives {'0': 0, '1': 1}
# So, class 0 is 'Nevus' and class 1 is 'Melanoma'
target_names = ['Nevus (0)', 'Melanoma (1)']
print(classification_report(y_true, y_pred_classes, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix on Validation Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve and AUC Score
auc_score = roc_auc_score(y_true, y_pred_proba)
print(f"ROC AUC Score (calculated from predictions): {auc_score:.4f}")

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    auc = history.history.get('auc', []) # Use .get for safety if 'auc' metric name varies
    val_auc = history.history.get('val_auc', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    if auc and val_auc:
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, auc, label='Training AUC')
        plt.plot(epochs_range, val_auc, label='Validation AUC')
        plt.legend(loc='lower right')
        plt.title('Training and Validation AUC')

    plt.show()

plot_training_history(history)

print("Script finished.")