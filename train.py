# python train.py --dataset data --model lang.model --labelbin mlb.pickle

	# extract set of class labels from the image path and update the
	# labels list
	# l = label = imagePath.split(os.path.sep)[-2].split("_")



# set the matplotlib backend so figures can be saved in the background
import matplotlib
import datetime
# matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.basicnet import BasicNet
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import random
import pickle
import cv2
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 15
INIT_LR = 1e-3
BS = 2
IMAGE_DIMS = (48, 48, 3)

# disable eager execution
tf.compat.v1.disable_eager_execution()

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    a = imagePath.split('\\')
    a = a[1]
    l = label = a
    labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("Labels: " ,labels)
print("[INFO] class labels:")
labelbinarizer = LabelBinarizer()
labels = labelbinarizer.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(labelbinarizer.classes_):
	print("{}. {}".format(i + 1, label))
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, random_state=42)

# # construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
# 	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# 	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
# model = SmallerVGGNet.build(
# 	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
# 	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
# 	finalAct="sigmoid")

model = BasicNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(labelbinarizer.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network...")
# H = model.fit(
# 	x=aug.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY),
# 	steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS, verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# fit network
# history = None
# for i in range(EPOCHS):
#     history = model.fit(trainX, trainY,validation_data=(testX, testY),callbacks=[tensorboard_callback],epochs=1, batch_size=BS, verbose=1, shuffle=True)
#     # history = model.fit(trainX, trainY,epochs=1, batch_size=BS, verbose=1, shuffle=True)
#     # model.reset_states()
history = model.fit(trainX, trainY,validation_data=(testX, testY),callbacks=[tensorboard_callback],epochs=EPOCHS, batch_size=BS, verbose=1, shuffle=True)


_, acc = model.evaluate(testX, testY, verbose=0)
print('> %.3f' % (acc * 100.0))


# H = model.fit(
# 	trainX, trainY, batch_size=BS,
# 	validation_data=(testX, testY),
# 	steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")
# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(labelbinarizer))
f.close()

# plot the training loss and accuracy
# list all data in history
print(history.history.keys())
# summarize history for accuracy
loss_train = history.history['loss']
train_accuracy = history.history['accuracy']

loss_val = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

print("Loss Train :",loss_train)
print("\n")
print("Loss loss :",loss_val)

titleA = 'Training and Validation loss - Train :' + str(loss_train[-1]) + "," + " Valid. :" + str(loss_val[-1])

epochs = range(EPOCHS)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title(titleA)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# plt.savefig('Training and Validation loss.png')


titleB = 'Training and Validation accuracy - Accuracy :' + str(train_accuracy[-1]) + "," + " Valid. :" + str(val_accuracy[-1])
epochs = range(EPOCHS)
plt.plot(epochs, train_accuracy, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
plt.title(titleB)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# plt.savefig('Training and Validation accuracy.png')