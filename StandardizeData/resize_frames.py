import cv2 as cv
import os

#change from 1280x720  to 720x405

dir_name = "NonFireVid6Frames"

file_names = os.listdir(dir_name)
#print file_names

scale = 0.5625

for name in file_names:
  print name
  path = dir_name + '/' + name
  curr_im = cv.imread(path)
  resized_im = cv.resize(curr_im, None, fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
  new_path = dir_name + '/' + name
  cv.imwrite(new_path, resized_im)
