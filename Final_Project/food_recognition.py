import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# TODO
# 1. download database from https://www.kaggle.com/kmader/food41/download
# 2. unzip to dir input

# constants
data_dir = './input/images/'
IMAGE_SHAPE = (224, 224)
ResNet_V2_50 = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5'
MobileNet_V3_100 = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5"
EfficientNet_V2_b0 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2"

# Prepare data
dataGenerator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = dataGenerator.flow_from_directory(
    data_dir,
    target_size=IMAGE_SHAPE,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = dataGenerator.flow_from_directory(
    data_dir,
    target_size=IMAGE_SHAPE,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# MobileNet V3 Model Building¶
model_MobileNet = tf.keras.Sequential([
    hub.KerasLayer(MobileNet_V3_100, input_shape=IMAGE_SHAPE+(3,), name="MobileNet_V3_100"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(101, activation='softmax', name='Output_layer')
])

model_MobileNet.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model_MobileNet.summary()
mobileNet_model = model_MobileNet.fit(train_data, epochs=10, verbose=1)
model_MobileNet.evaluate(val_data)


# EfficientNet V2 Model Building¶
model_EfficientNet = tf.keras.Sequential([
    hub.KerasLayer(EfficientNet_V2_b0, trainable=False, input_shape=IMAGE_SHAPE+(3,), name='EfficientNet_V2_b0'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(101, activation='softmax', name='Output_layer')
])

model_EfficientNet.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model_EfficientNet.summary()
efficientNet_model = model_EfficientNet.fit(train_data, epochs=10, verbose=1)
model_EfficientNet.evaluate(val_data)


# ResNet Model Building
model_ResNet = tf.keras.Sequential([
    hub.KerasLayer(ResNet_V2_50, trainable=False, input_shape=IMAGE_SHAPE+(3,), name='Resnet_V2_50'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(101, activation='softmax', name='Output_layer')
])

model_ResNet.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)


model_ResNet.summary()
resnet_model = model_ResNet.fit(train_data, epochs=10, verbose=1)
model_ResNet.evaluate(val_data)


# Compare models
def plot_graph(history, history_1, history_2):
    loss_mobile = history.history['loss']
    loss_ef = history_1.history['loss']
    loss_res = history_2.history['loss']

    Accuracy_mobile = history.history['accuracy']
    Accuracy_ef = history_1.history['accuracy']
    Accuracy_res = history_2.history['accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss_mobile, label='MobileNet Loss')
    plt.plot(epochs, loss_ef, label='EfficientNet Loss')
    plt.plot(epochs, loss_res, label='ResNet Loss')
    plt.title('Epochs Vs Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, Accuracy_mobile, label='MobileNet Accuracy')
    plt.plot(epochs, Accuracy_ef, label='EfficientNet Accuracy')
    plt.plot(epochs, Accuracy_res, label='ResNet Accuracy')
    plt.title('Epochs Vs Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()


plot_graph(mobileNet_model, efficientNet_model, resnet_model)