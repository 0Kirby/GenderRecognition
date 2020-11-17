# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Helper libraries
from pandas import read_csv
import numpy as np
import pandas as pd
import cv2

print(tf.__version__)

array_of_img = []  # this if for store all of the image data
image_size = 100

directory_name = "train/train/"
for i in range(1, 18001):
    # print(filename) #just for test
    # img is used to store the image data
    # img = cv2.imread(directory_name + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(directory_name + str(i) + ".jpg")
    img = img / 255.0
    img = cv2.resize(img, (image_size, image_size))
    array_of_img.append(img)
train_images = np.array(array_of_img)
array_of_img = []

dataframe = read_csv('train.csv')
array = dataframe.values
train_labels = np.array(array[:, 1], dtype='int8')
del dataframe
del array
class_names = ['male', 'female']

# train_images = train_images.reshape(train_images.shape[0], image_size, image_size, 1)
# train_images = train_images.astype('float32')

X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=3)

data_augmentation = keras.Sequential(
    [
        keras.layers.GaussianNoise(0.1, input_shape=(image_size, image_size, 3)),
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)


model = keras.Sequential([
    data_augmentation,
    keras.layers.Conv2D(25, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(200, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation='softmax')
])

model.summary()




model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )

save_weights = 'save_weights.h5'
last_weights = 'last_weights.h5'
best_weights = 'best_weights.h5'

# model.load_weights(save_weights)

checkpoint = keras.callbacks.ModelCheckpoint(best_weights, monitor='val_accuracy', save_best_only=True, mode='max',
                                             verbose=1)
reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto',
                                           min_delta=0.0001, cooldown=0, min_lr=0)
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
callbacks = [checkpoint]


# hist = model.fit(train_images, train_labels, epochs=100)
hist = model.fit(X_train, Y_train, epochs=2000, validation_data=(X_val, Y_val), use_multiprocessing=True,
                 callbacks=callbacks, workers=3)

model.save_weights(last_weights)
plt.figure()

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')  # 'bo'为画蓝色圆点，不连线
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation accuracy,Training and validation loss')
plt.legend()  # 绘制图例，默认在右上角

plt.show()

# model.load_weights(best_weights) 加载本次训练最佳权重
model.load_weights(save_weights) # 加载之前已保存的能复现结果的最佳权重
del train_images
del train_labels

directory_name = "test/test/"

for i in range(18001, 23709):
    # print(filename) #just for test
    # img is used to store the image data
    # img = cv2.imread(directory_name + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(directory_name + str(i) + ".jpg")
    img = img / 255.0
    img = cv2.resize(img, (image_size, image_size))
    array_of_img.append(img)
test_images = np.array(array_of_img)
del array_of_img
# test_images = test_images.reshape(test_images.shape[0], image_size, image_size, 1)
# test_images = test_images.astype('float32')
# probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)

predictions = model.predict(test_images)
results = np.argmax(predictions, axis=1)
submissions = pd.read_csv('test.csv')
submissions['label'] = results
submissions.to_csv('submission.csv', index=False)
