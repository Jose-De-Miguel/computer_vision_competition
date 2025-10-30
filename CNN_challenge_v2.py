# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:19:29 2025

@author: chdem
"""

import uuid
import numpy as np
import warnings
import rasterio
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# The rest of your imports
from keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LeakyReLU,
    Dropout, LayerNormalization, Input, ZeroPadding2D, Activation,
    AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Concatenate,
    RandomTranslation
)
from keras.optimizers import SGD, Adam, RMSprop
from keras.optimizers.schedules import PiecewiseConstantDecay
from keras.initializers import he_normal, VarianceScaling, Orthogonal, GlorotNormal
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

# Métricas sklearn
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_fscore_support,
    top_k_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay
)


class GenericObject:
    """
    Generic object data.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.bb = (-1, -1, -1, -1)
        self.category= -1
        self.score = -1

class GenericImage:
    """
    Generic image data.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.objects = list([])

    def add_object(self, obj: GenericObject):
        self.objects.append(obj)
        
        
categories = {0: 'Cargo plane', 1: 'Small car', 2: 'Bus', 3: 'Truck', 4: 'Motorboat', 5: 'Fishing vessel', 6: 'Dump truck', 7: 'Excavator', 8: 'Building', 9: 'Helipad', 10: 'Storage tank', 11: 'Shipping container', 12: 'Pylon'}

"""
# DESCOMENTAR PARA DESCOMPRIMIR ZIP
# Extraer cada ZIP
os.makedirs(extract_base_path, exist_ok=True)

for zip_name in zip_files:
    zip_path = os.path.join(zip_folder, zip_name)
    extract_path = os.path.join(extract_base_path, zip_name.replace('.zip', ''))
    os.makedirs(extract_path, exist_ok=True)

    print(f'Extrayendo {zip_name} en {extract_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
"""
print("✅ Todo ok.")

def load_geoimage(filename):
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
    src_raster = rasterio.open(r'.\vision_descomprimido\xview_recognition\\' + filename, 'r')
    # RasterIO to OpenCV (see inconsistencies between libjpeg and libjpeg-turbo)
    input_type = src_raster.profile['dtype']
    input_channels = src_raster.count
    img = np.zeros((src_raster.height, src_raster.width, src_raster.count), dtype=input_type)
    for band in range(input_channels):
        img[:, :, band] = src_raster.read(band+1)
    return img

def generator_images(objs, batch_size, do_shuffle=False):
    while True:
        if do_shuffle:
            np.random.shuffle(objs)
        groups = [objs[i:i+batch_size] for i in range(0, len(objs), batch_size)]
        for group in groups:
            images, labels = [], []
            for (filename, obj) in group:
                # Load image
                images.append(load_geoimage(filename))
                probabilities = np.zeros(len(categories))
                probabilities[list(categories.values()).index(obj.category)] = 1
                labels.append(probabilities)
            images = np.array(images).astype(np.float32)
            labels = np.array(labels).astype(np.float32)
            yield images, labels


# Load database
json_file = r'.\vision_descomprimido\xview_recognition\xview_ann_train.json'
with open(json_file) as ifs:
    json_data = json.load(ifs)
ifs.close()



counts = dict.fromkeys(categories.values(), 0)
anns = []
for json_img, json_ann in zip(json_data['images'].values(), json_data['annotations'].values()):
    image = GenericImage(json_img['filename'])
    image.tile = np.array([0, 0, json_img['width'], json_img['height']])
    obj = GenericObject()
    obj.bb = (int(json_ann['bbox'][0]), int(json_ann['bbox'][1]), int(json_ann['bbox'][2]), int(json_ann['bbox'][3]))
    obj.category = json_ann['category_id']
    # Resampling strategy to reduce training time
    counts[obj.category] += 1
    image.add_object(obj)
    anns.append(image)
print(counts)



anns_train, anns_valid = train_test_split(anns, test_size=0.1, random_state=1, shuffle=True)
print('Number of training images: ' + str(len(anns_train)))
print('Number of validation images: ' + str(len(anns_valid)))

# Generate the list of objects from annotations
objs_train = [(ann.filename, obj) for ann in anns_train for obj in ann.objects]
objs_valid = [(ann.filename, obj) for ann in anns_valid for obj in ann.objects]


# Generators
batch_size = 16
train_generator = generator_images(objs_train, batch_size, do_shuffle=True)
valid_generator = generator_images(objs_valid, batch_size, do_shuffle=False)

#This block only checks that the immages have been loading correctly
try:
    L = len(train_generator)  # works for Sequence / ImageDataGenerator / tf.data via .__len__
    print("len(train_generator) =", L)
except TypeError:
    print("len(train_generator) unavailable (plain Python generator?)")

# 2) Try to actually get one batch
it_train = iter(train_generator)
try:
    first = next(it_train)
    assert isinstance(first, tuple), f"Generator must yield a tuple, got {type(first)}"
    print("First train batch shapes:",
          [getattr(x, "shape", type(x)) for x in (first if isinstance(first, tuple) else (first,))])
except StopIteration:
    raise RuntimeError("train_generator yielded 0 batches. Check your data and generator logic.")

# 3) Do the same for validation
it_valid = iter(valid_generator)
try:
    first_v = next(it_valid)
    assert isinstance(first_v, tuple), f"Validation generator must yield a tuple, got {type(first_v)}"
    print("First valid batch shapes:",
          [getattr(x, "shape", type(x)) for x in (first_v if isinstance(first_v, tuple) else (first_v,))])
except StopIteration:
    raise RuntimeError("valid_generator yielded 0 batches. Check your data and generator logic.")
    

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])



NUM_CLASSES = 13
BATCH_SIZE = 64
input_shape=(224,224,3)

inputs = layers.Input(shape=input_shape)

x = data_augmentation(inputs)

# Bloque 1
x = layers.Conv2D(32, (3,3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, (3,3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.2)(x)

# Bloque 2
x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.3)(x)

# Bloque 3
x = layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.4)(x)

# Bloque 4tras
x = layers.Conv2D(256, (3,3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.4)(x)

# Cabeza
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

print("Exec point 2")

model = models.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,               # <- aquí va la paciencia
    restore_best_weights=True,

)


history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    #class_weight=dict(enumerate(class_weights))
)

