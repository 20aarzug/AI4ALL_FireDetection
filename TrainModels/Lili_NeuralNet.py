import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import skimage
import os
from skimage import io
from sklearn.externals import joblib

forest_fire_names = os.listdir("ForestFire/Fire1Resized")

im = skimage.io.imread("ForestFire/Fire1Resized/" + forest_fire_names[0])
print type(im)
print im.shape

non_fire_names = os.listdir("NonFireResized/NonFireVid2FramesSmall")

num_images = len(forest_fire_names) + len(non_fire_names)

#print forest_fire_names
#print non_fire_names

data = np.ndarray(shape=(num_images,im.shape[0], im.shape[1],im.shape[2]))
print data.shape

c = 0
for file_name in forest_fire_names:
  temp_im = skimage.io.imread("ForestFire/Fire1Resized/" + file_name)
  data[c] = temp_im
  c += 1

for file_name in non_fire_names:
  temp_im = skimage.io.imread("NonFireResized/NonFireVid2FramesSmall/" + file_name)
  data[c] = temp_im
  c += 1

labels = []
for i in range(len(forest_fire_names)):
  labels.append(1)

for i in range(len(non_fire_names)):
  labels.append(0)

print len(data)
data_2d = data.reshape(len(data),-1)
print data_2d.shape

#train, test, train_labels, test_labels = train_test_split(data_2d, labels, test_size = 0.2, random_state = 42)

mlp = MLPClassifier(alpha=1)
model = mlp.fit(data_2d, labels)

filename = 'model_fire1_nonfire2.sav'
joblib.dump(model, filename)

test_file_names = os.listdir("NonFireResized/NonFireVid3FramesSmall")
im2 = skimage.io.imread("NonFireResized/NonFireVid3FramesSmall/" + test_file_names[0])
num_im = len(test_file_names)
test = np.ndarray(shape=(num_im, im2.shape[0], im.shape[1], im.shape[2]))
test_2d = test.reshape(len(test),-1)
test_labels = []
for i in range(num_im):
  test_labels.append(0)

preds = mlp.predict(test_2d)
print(preds)
print(accuracy_score(test_labels,preds))
