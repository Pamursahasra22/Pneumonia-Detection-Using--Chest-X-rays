import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TF noise during training

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Configuration
img_size = 150
labels = ['PNEUMONIA', 'NORMAL'] 

def get_data(data_dir):
    data = []
    # Check common paths for Kaggle or local structures
    base = ""
    if os.path.exists('./chest_xray'): base = './chest_xray/'
    
    full_path = os.path.join(base, data_dir)
    print(f"Processing: {full_path}")
    
    for label in labels:
        path = os.path.join(full_path, label)
        class_num = labels.index(label)
        if not os.path.exists(path): continue
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except: continue
    return data

def preprocess(data):
    if not data: return np.array([]), np.array([])
    x, y = [], []
    for f, l in data:
        x.append(f); y.append(l)
    return np.array(x).reshape(-1, img_size, img_size, 1) / 255.0, np.array(y)

# 1. Load Data
x_train, y_train = preprocess(get_data('train'))
x_val, y_val = preprocess(get_data('val'))

# 2. Build Model
model = models.Sequential([
    layers.Input(shape=(150, 150, 1)),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary Output
])

model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.1, horizontal_flip=True)
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.3)

print("Starting Training...")
model.fit(datagen.flow(x_train, y_train, batch_size=32), 
          epochs=5, validation_data=(x_val, y_val), callbacks=[lr_reduction])

# 4. Save
model.save('pneumonia_model.h5')
print("✅ Model saved as 'pneumonia_model.h5'")