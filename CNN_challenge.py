

# Assistant
# Import ImageDataGenerator from tensorflow instead of keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Numpy
import numpy as np

import tensorflow as tf
import random
import math
import keras
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# The rest of your imports
from keras.models import Sequential, Model
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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

# Imagen / procesamiento
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import rotate

# Utilidades
import os
import zipfile  # Added missing import
from typing import Dict, List, Tuple
from tqdm.notebook import tqdm


# Librerías para visualización de datos
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.image as mpimg

#pandas
import pandas as pd



zip_folder = './'
zip_files = ['xview_recognition.zip']
extract_base_path = './vision_descomprimido'

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



### **Carga datos train**





# Ruta principal del dataset
ruta_train = "./vision_descomprimido/xview_recognition/xview_train"

# Obtener todas las clases (carpetas dentro de xview_train)
clases = [nombre for nombre in os.listdir(ruta_train) if os.path.isdir(os.path.join(ruta_train, nombre))]

etiquetas_train = []
file_list = []
num_imagest = []

# Recorremos cada clase
for clase in clases:
    # Ruta de la carpeta de la clase
    folder_path = os.path.join(ruta_train, clase)

    # Lista de archivos en esa carpeta
    aux1 = os.listdir(folder_path)

    # Cantidad de imágenes
    cantidad_img = len(aux1)

    # Etiquetas repetidas según número de imágenes
    aux = [clase] * cantidad_img

    # Guardamos resultados
    num_imagest.append(cantidad_img)
    etiquetas_train += aux
    file_list += aux1

print("Clases encontradas:", clases)
print("Número de imágenes por clase:", num_imagest)
print("Total imágenes:", sum(num_imagest))


# ======================
# Ahora cargamos las imágenes en arrays
# ======================

data_images_train = []

for i in range(len(file_list)):

    # Ruta completa de la imagen
    image_path = os.path.join(ruta_train, etiquetas_train[i], file_list[i])


    # Abrimos y convertimos a array



    try:
        with Image.open(image_path) as img:
            img.verify()  # Solo comprueba, no carga toda la imagen
            image_array = np.array(img)
    
    except Exception as e:
        print("Error al verificar la imagen:", e)


   

    data_images_train.append(image_array)



print("Número total de arrays cargados:", len(data_images_train))

### **Carga datos test**



# Ruta principal del dataset
ruta_test = "./vision_descomprimido/xview_recognition/xview_test"

# Obtener todas las clases (carpetas dentro de xview_train)
clases = [nombre for nombre in os.listdir(ruta_test) if os.path.isdir(os.path.join(ruta_test, nombre))]

etiquetas_test = []
file_list = []
num_images = []

# Recorremos cada clase
for clase in clases:
    # Ruta de la carpeta de la clase
    folder_path = os.path.join(ruta_test, clase)

    # Lista de archivos en esa carpeta
    aux1 = os.listdir(folder_path)

    # Cantidad de imágenes
    cantidad_img = len(aux1)

    # Etiquetas repetidas según número de imágenes
    aux = [clase] * cantidad_img

    # Guardamos resultados
    num_images.append(cantidad_img)
    etiquetas_test += aux
    file_list += aux1



# ======================
# Ahora cargamos las imágenes en arrays
# ======================

data_images_test = []

for i in range(len(file_list)):
    # Ruta completa de la imagen
    image_path = os.path.join(ruta_test, etiquetas_test[i], file_list[i])


    try:
        with Image.open(image_path) as img:
            img.verify()  # Solo comprueba, no carga toda la imagen
            image_array = np.array(img)
    
    except Exception as e:
        print("Error al verificar la imagen:", e)

    data_images_test.append(image_array)





# Crear DataFrame con las imágenes y sus etiquetas
df = pd.DataFrame({
    'imagen': data_images_train,
    'etiqueta': etiquetas_train
})

# Ordenar por etiqueta
df_sorted = df.sort_values(by='etiqueta').reset_index(drop=True)



imagenes_ordenadas = df_sorted['imagen'].tolist()
etiquetas_ordenadas = df_sorted['etiqueta'].tolist()


print("Ok 2")
# Crear DataFrame con las imágenes y sus etiquetas
df2 = pd.DataFrame({
    'imagen': data_images_test,
    'etiqueta': etiquetas_test
})

# Ordenar por etiqueta
df_sorted2 = df2.sort_values(by='etiqueta').reset_index(drop=True)


data_images_test_ord = df_sorted2['imagen'].tolist()
etiquetas_test_ord = df_sorted2['etiqueta'].tolist()


clases_unicas = np.unique(etiquetas_test_ord)


del df, df_sorted, df2#, df_sorted2

indices = []
etiquetas_unicas = np.unique(etiquetas_ordenadas)
count = 1
plt.figure(figsize=(18, 40))
for etiqueta in etiquetas_unicas:
    # Encuentra el índice de la primera ocurrencia de la etiqueta
    indice = np.where(np.array(etiqueta) == etiquetas_ordenadas)[0][0]
    indices.append(indice)

    

    count = count + 1



etiquetas = np.array(etiquetas_ordenadas)
indices = [0]  # el primer índice siempre empieza en 0

for i in range(1, len(etiquetas)):
    if etiquetas[i] != etiquetas[i-1]:
        indices.append(i)

print("Indices de cambio de clase:", indices)

nimg = np.array(indices[1:] + [len(etiquetas)]) - np.array(indices)
indices = np.array(indices + [len(etiquetas)])  # cerrar el rango
print("Total de imágenes por clase:", nimg)

nimg = np.round(nimg*0.2)
print(nimg) # 20% del total de imagenes

# Partimos de listas vacías
train_images, train_labels = [], []
validation_images, validation_labels = [], []

# Supongo que:
# - imagenes_ordenadas y etiquetas_ordenadas están alineadas y ordenadas por clase
# - etiquetas_unicas = df_sorted['etiqueta'].unique()
# - indices tiene los inicios de cada clase y un último índice de cierre (len(etiquetas_ordenadas))
# - nimg[i] = round(0.2 * total_imagenes_de_la_clase_i)

for i in range(len(etiquetas_unicas)):
    start = indices[i]
    end   = indices[i+1]           # cierre de la clase i
    k     = int(nimg[i])           # 20% para validación (ya redondeado)

    # --- VALIDACIÓN (20%) ---
    val_imgs = imagenes_ordenadas[start : start + k]
    val_labs = [etiquetas_unicas[i]] * len(val_imgs)
    validation_images += val_imgs
    validation_labels += val_labs   # <- CORREGIDO (antes pisabas con validation_images)

    # --- TRAIN (resto 80%) ---
    tr_imgs = imagenes_ordenadas[start + k : end]
    tr_labs = [etiquetas_unicas[i]] * len(tr_imgs)
    train_images += tr_imgs
    train_labels += tr_labs

# Test viene de fuera (otro split/dataset)
test_images = data_images_test_ord
test_labels = etiquetas_test_ord

# Comprobaciones rápidas
print(f"Train: {len(train_images)} imágenes, {len(train_labels)} etiquetas")
print(f"Val  : {len(validation_images)} imágenes, {len(validation_labels)} etiquetas")
print(f"Test : {len(test_images)} imágenes, {len(test_labels)} etiquetas")



train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)
test_labels = np.array(test_labels)









# Se convierten las etiquetas de texto en representaciones numericas
train_labels = LabelEncoder().fit_transform(train_labels)
validation_labels = LabelEncoder().fit_transform(validation_labels)
test_labels = LabelEncoder().fit_transform(test_labels)

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_labels),
                                     y=train_labels)

def aleatorio(train_images, train_labels):
    ale = list(zip(train_images, train_labels))
    np.random.shuffle(ale)
    train_images, train_labels = zip(*ale)
    return np.array(train_images), np.array(train_labels)

train_images, train_labels = aleatorio(train_images, train_labels)
train_labels = to_categorical(train_labels)

validation_images, validation_labels = aleatorio(validation_images, validation_labels)
validation_labels = to_categorical(validation_labels)

test_images, test_labels = aleatorio(test_images, test_labels)
test_labels = to_categorical(test_labels)


print("Exec point 1")

from tensorflow.keras import layers, models

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



# In[2]:


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
    train_images, train_labels,
    validation_data=(validation_images, validation_labels),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=dict(enumerate(class_weights))
)




