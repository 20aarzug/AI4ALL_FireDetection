from sklearn import svm, datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

filename_list = os.listdir("Data/FireDump")
#filename_list3 = os.listdir("Data/Fire3Resized")
# add other 3 fire files
filename_list2 = os.listdir("Data/NonFireDump")
#filename_list4 = os.listdir("Data/NonFireVid4FramesSmall")
#add other 3 non-fire files

#add campfire files

filename = filename_list[0]
image = skimage.io.imread("Data/FireDump/" + filename)

num_examples = len(filename_list)

#num_examples3 = len(filename_list3)

num_examples2 = len(filename_list2)

#num_examples4 = len(filename_list4)

total = num_examples + num_examples2 # + num_examples3 + num_examples4

data = np.ndarray(shape=(total,image.shape[0],image.shape[1],image.shape[2]))
i = 0
for filename in filename_list:
    image = skimage.io.imread("Data/FireDump/" + filename)
    data[i] = image
    i+=1

#for filename3 in filename_list3:
    #image3 = skimage.io.imread("Data/Fire3Resized/" + filename3)
    #data[i] = image3
    #i+=1

for filename2 in filename_list2:
    image2 = skimage.io.imread("Data/NonFireDump/" + filename2)
    data[i] = image2
    i+=1

#for filename4 in filename_list4:
    #image4 = skimage.io.imread("Data/NonFireVid4FramesSmall/" + filename4)
    #data[i] = image4
    #i+=1

fire = 1
non_fire = 0
lst = range(num_examples)
#lst3 = range(num_examples3)
lst2 = range(num_examples2) 
#lst4 = range(num_examples4)
labels = []
for x in lst:
  labels.append(fire)
#for x in lst3:
  #labels.append(fire)
for x in lst2:
  labels.append(non_fire)
#for x in lst4:
  #labels.append(non_fire)

print (labels)
print (len(labels))
print (data.shape)
a
num_samples = len(data)
data_2d = data.reshape(num_samples,-1) #multiplies all three dimensions into 1 
data_2d.shape

train, test, train_labels, test_labels = train_test_split(data_2d,labels,test_size=0.2,random_state=42)

print(len(train))
print(len(test))
print(len(train_labels))
print(len(test_labels))

print(test_labels)

clsf = svm.SVC()
print ("Model training:")
model = clsf.fit(train, train_labels)

preds = model.predict(test)
print(preds)
print(accuracy_score(test_labels,preds))
print(precision_score(test_labels,preds))
print(recall_score(test_labels,preds))
print(f1_score(test_labels,preds))
