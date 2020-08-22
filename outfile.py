# ============================IMPORTANT=====================================
# This file purpose is to read the saved pickle files from the directory 
# The format for this file is as follows
# python3 outfile.py main_file parts_file
# Details of Parts file:
#0th Location: Lips
#1st Location: Lip Line
#2nd Location: Left Eyebrows
#3rd Location: Right Eyebrows
#4th Location: Left Eye
#5th Location: Right Eye
#6th Location: Nose
#7th Location: Jaw Line
#8th Location: ForeHead
#9th Location: Left Teardrop
#10th Location: Right Teardrop
# ==========================================================================

import pickle 
import sys
import cv2

main_image_set = sys.argv[1]
parts_image_set = sys.argv[2]

infile_1 = open(main_image_set,'rb')
frame = pickle.load(infile_1)
infile_1.close()
print('Main Frame Lenght:{}'.format(len(frame)))

infile_2 = open(parts_image_set,'rb')
final_roi = pickle.load(infile_2)
infile_2.close()
print('Parts Frame Lenght:{}'.format(len(final_roi)))

for i in range(len(frame)):
	cv2.imshow("Frame", frame[i])
	cv2.imwrite('Frame.jpg',frame[i])
	cv2.imshow('Lips',final_roi[i][0])
	cv2.imwrite('Lips.jpg',final_roi[i][0])			
	cv2.imshow('Lip Line',final_roi[i][1])
	cv2.imwrite('Lip Line.jpg',final_roi[i][1])			
	cv2.imshow('Left Eyebrows',final_roi[i][2])
	cv2.imwrite('Left Eyebrows.jpg',final_roi[i][2])			
	cv2.imshow('right Eyebrows',final_roi[i][3])
	cv2.imwrite('right Eyebrows.jpg',final_roi[i][3])			
	cv2.imshow('Left Eye',final_roi[i][4])	
	cv2.imwrite('Left Eye.jpg',final_roi[i][4])		
	cv2.imshow('Right Eye',final_roi[i][5])	
	cv2.imwrite('Right Eye.jpg',final_roi[i][5])		
	cv2.imshow('Nose',final_roi[i][6])
	cv2.imwrite('Nose.jpg',final_roi[i][6])			
	cv2.imshow('Jaw Line',final_roi[i][7])
	cv2.imwrite('Jaw Line.jpg',final_roi[i][7])			
	cv2.imshow('ForeHead',final_roi[i][8])	
	cv2.imwrite('ForeHead.jpg',final_roi[i][8])		
	cv2.imshow('Left Teardrop',final_roi[i][9])
	cv2.imwrite('Left Teardrop.jpg',final_roi[i][9])			
	cv2.imshow('Right Teardrop',final_roi[i][10])
	cv2.imwrite('Right Teardrop.jpg',final_roi[i][10])
	key = cv2.waitKey(50) & 0xFF
	if key == ord("q"):
		break
