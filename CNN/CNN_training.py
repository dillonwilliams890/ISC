#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Rescaling
from sklearn.metrics import confusion_matrix, classification_report
import pathlib
from tensorflow.keras.optimizers import Adam


#%%
train_data_dir='dataset_mask'
test_data_dir='data'
img_height=100
img_width=100
batch_size=32


train_ds = tf.keras.utils.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.2,
  subset="training",
  class_names=None,
  color_mode='rgb',
  seed=1337,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.2,
  subset="validation",
  class_names=None,
  color_mode='rgb',
  seed=1337,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_data_dir,
  seed=1337,
  class_names=None,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size)
# %%
class_names = train_ds.class_names


# Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#%%
dnn_model = Sequential()

imported_model= tf.keras.applications.ResNet50(include_top=False,
input_shape=(100,100,3),
pooling='avg',classes=2,
weights='imagenet')
for layer in imported_model.layers:
    layer.trainable=False

dnn_model.add(tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input, input_shape=(100, 100, 3)))
dnn_model.add(imported_model)
dnn_model.add(BatchNormalization())
dnn_model.add(Flatten())
dnn_model.add(Dense(512, activation='relu'))
dnn_model.add(Dense(2, activation='softmax'))

dnn_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# history_dnn=dnn_model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=10
# )


monitor = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=2, verbose=1, mode='auto',
        restore_best_weights=True)

history_dnn = dnn_model.fit(train_ds, batch_size=batch_size, epochs=10,
                      verbose=2, validation_data=val_ds, callbacks=[monitor])

plt.plot(history_dnn.history['accuracy'], label='accuracy')
plt.plot(history_dnn.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#%%
predictions = np.array([])
labels =  np.array([])
for x, y in test_ds:
  predictions = np.concatenate([predictions, np.argmax(dnn_model.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=0)])

# predictions = dnn_model.predict(test_ds)
# y_predictions = np.argmax (predictions, axis = 1)
result = confusion_matrix(labels, predictions , normalize='pred')

#%%
y_pred = []
y_true = []
names=[]
threshold=0.97
# iterate over the dataset
for image_batch, label_batch in test_ds:   # use dataset.unbatch() with repeat
#    names.append(test_ds.filepaths[index])
   # append true labels
   y_true.append(label_batch)
   # compute predictions
   preds = dnn_model.predict(image_batch)
   # append predicted labels
   y_pred.append(np.where(preds > threshold, 1, 0))
#    for i in range(len(preds)):
#         predict=(np.where(preds[i] > threshold, 1, 0))[1]
#         if abs(label_batch[i]- predict)>0:
#             plt.figure(figsize=(10, 10))
#             plt.imshow(image_batch[0].numpy().astype("uint8"))
#             plt.title(class_names[label_batch[i]])
#             plt.text(3.5, 0.9, class_names[predict], fontsize = 23)
#             # plt.text(0, -9, names[i], fontsize = 23)

# convert the true and predicted labels into tensors
true_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_labels, predicted_labels[:,1],normalize='pred')
print(cm)
sns.heatmap(cm, annot=True)
# %%
results= np.absolute(true_labels-predicted_labels[:,1])
ind=np.where(results>0)[0]
names=test_ds.file_paths
testunbatch_ds = test_ds.unbatch()
images = list(testunbatch_ds.map(lambda x, y: x))
labels = list(testunbatch_ds.map(lambda x, y: y))

for i in range(len(ind)):
            plt.figure(figsize=(10, 10))
            plt.imshow(images[ind[i]].numpy().astype("uint8"))
            plt.title(class_names[true_labels[ind[i]]])
            plt.text(3.5, 0.9, class_names[predicted_labels[ind[i],1]], fontsize = 23)
            plt.text(0, -9, names[ind[i]], fontsize = 23)
# %%
