#=============================================================== Functions Import
from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from imutils import face_utils
import imutils
import dlib
import copy 
from matplotlib import pyplot as plt
import pickle
import sys

#====================================================================Functions

# Extracting Facial Features from Face Images like Nose, Forehead, Cheecks(Teardrop) etc. 
def new_features(shape,face_utils):
    #Forehead
    a= [shape[19,0],shape[19,1]-10]
    b= [shape[21,0],shape[19,1]-10]
    c= [shape[22,0],shape[19,1]-10]
    d =[shape[24,0],shape[19,1]-10]
    e = [shape[24,0],shape[19,1]-20]
    f = [shape[24,0],shape[19,1]-30]
    g = [shape[22,0],shape[19,1]-30]
    h = [shape[21,0],shape[19,1]-30]
    i = [shape[19,0],shape[19,1]-30]
    j = [shape[19,0],shape[19,1]-20]

    new_shape = np.concatenate((shape,np.array(a).reshape((-1,2)),
np.array(b).reshape((-1,2)),np.array(c).reshape((-1,2)),
np.array(d).reshape((-1,2)),np.array(e).reshape((-1,2)),np.array(f).reshape((-1,2)),
np.array(g).reshape((-1,2)),np.array(h).reshape((-1,2)),np.array(i).reshape((-1,2)),
np.array(j).reshape((-1,2))),axis = 0)

    #left tear Drop
    k = [shape[36,0],shape[3,1]]
    l = [shape[39,0],shape[3,1]]
    m = [shape[39,0],shape[1,1]]
    n = [shape[36,0],shape[1,1]]

    #right tear Drop
    o = [shape[42,0],shape[3,1]]
    p = [shape[45,0],shape[3,1]]
    q = [shape[45,0],shape[1,1]]
    r = [shape[42,0],shape[1,1]]

    new_shape = np.concatenate((new_shape,np.array(k).reshape((-1,2)),
    np.array(l).reshape((-1,2)),np.array(m).reshape((-1,2)),np.array(n).reshape((-1,2)),
    np.array(o).reshape((-1,2)),np.array(p).reshape((-1,2)),np.array(q).reshape((-1,2)),
    np.array(r).reshape((-1,2))))


    features_list = []

    for (name, (i, j)) in face_utils:
    #     print(name)
        features_list.append(np.array(name))
        features_list.append(np.array(i))
        features_list.append(np.array(j))
    features_list.append('forehead')
    features_list.append(68)
    features_list.append(78)
    features_list.append('left tear drop area')
    features_list.append(78)
    features_list.append(82)
    features_list.append('right tear drop area')
    features_list.append(82)
    features_list.append(86)
    features_list = np.array(features_list).reshape((-1,3))

    new_facial_landmarks = []
    for i in features_list:
        v = []
        v.append(i[0])
        f=[]
        f.append(i[1])
        f.append(i[2])
        v.append(tuple(f))
        new_facial_landmarks.append(tuple(v))
    return(new_shape, new_facial_landmarks)
    
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('{}'.format(sys.argv[1]))

print("[INFO] setting up the video...")
vs = cv2.VideoCapture('{}'.format(sys.argv[2]))
time.sleep(3.0)

(h, w) = (None, None)
zeros = None

f_frame = []
f1_frame = []
try:
	while True:

		ret,frame = vs.read()
		
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 1)
		final_roi = []
		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			v = face_utils.FACIAL_LANDMARKS_IDXS.items()
			ns,nfl = new_features(shape,v)
			img = copy.deepcopy(frame)
			for (x, y) in ns:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			for (name,(i,j)) in nfl:
				clone = copy.deepcopy(frame)
				cv2.putText(clone, name, (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, 
				(255, 255, 255), 2)
				for (x, y) in ns[int(i):int(j)]:
					cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
					(x, y, w, h) = cv2.boundingRect(ns[int(i):int(j)])
				final_roi.append(img[y:y + h, x:x + w])
		cv2.imshow("Frame", frame)
		cv2.imshow('Lips',final_roi[0])			
		cv2.imshow('Lip Line',final_roi[1])			
		cv2.imshow('Left Eyebrows',final_roi[2])			
		cv2.imshow('right Eyebrows',final_roi[3])			
		cv2.imshow('Left Eye',final_roi[4])			
		cv2.imshow('Right Eye',final_roi[5])			
		cv2.imshow('Nose',final_roi[6])			
		cv2.imshow('Jaw Line',final_roi[7])			
		cv2.imshow('ForeHead',final_roi[8])			
		cv2.imshow('Left Teardrop',final_roi[9])			
		cv2.imshow('Right Teardrop',final_roi[10])			
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break
		f_frame.append(frame)
		f1_frame.append(final_roi)
except:
	pass

print("[INFO] cleaning up...")
f1_name = 'parts_file'
of_1 = open(f1_name,'wb')
pickle.dump(f1_frame,of_1)
of_1.close()
print('Lenght of the Parts file: {}'.format(len(f1_frame)))

f2_name = 'main_file'
of_2 = open(f2_name,'wb')
pickle.dump(f_frame,of_2)
of_2.close()
print('Length of the Image Frame File: {}'.format(len(f_frame)))
cv2.destroyAllWindows()





