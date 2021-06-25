import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
import cv2
import numpy as np
#import pickle

CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
DATADIR = "Dataset"
training_data = []

# =============================================================================
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.333)
# sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
# =============================================================================

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (28, 28))
        training_data.append([new_array, class_num])

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 28, 28, 1)
y = np.array(y).astype(float)

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(26))
model.add(Activation('softmax'))

model.compile(
        optimizer=tf.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

model.fit(X, y, batch_size=224, epochs=5) #, callbacks = [tensorboard])

result = []
for i in range(1, 27):
    image = cv2.imread('data/'+str(i)+'.png', 0)
    image = cv2.resize(image, (28,28))
    image = image.reshape(-1, 28, 28, 1).astype(float)
    prediction = model.predict_classes(image)
    result.append([CATEGORIES[i-1], CATEGORIES[prediction[0]]])

count = 0
for i, j in result:
    if i == j:
        count += 1
print(count)

# tf.keras.models.save_model(
#     model,
#     './trained_model1acc20.h5',
#     overwrite=True,
#     include_optimizer=True
# )