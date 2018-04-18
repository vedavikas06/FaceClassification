import numpy as np
import cv2
import csv
import os
import pandas
path = "/home/kaiser17/Desktop/REBELLION/project1/KenyanFemales/"
listoflists = []

 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
t=0
for filename in os.listdir(path):
	img= cv2.imread("/home/kaiser17/Desktop/REBELLION/project1/KenyanFemales/"+filename)
	 
	if img is None:
		continue
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	 
	gray=img
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	for (x,y,w,h) in faces:
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		if img is None:
			continue
		crop_img = img[y:y+h, x:x+w]
		crop_img = cv2.resize(crop_img, (100,100)) 
		uy=crop_img
		cf=np.array(uy)
		flat_arr = cf.ravel()
		 
		a_list=np.ndarray.tolist(flat_arr)
		 
		 
		listoflists.append(a_list)
		 
		t=t+1
		cv2.imwrite('/home/kaiser17/Desktop/REBELLION/project1/nigg/kingzz'+str(t)+'.jpg',crop_img)
		roi_gray = gray[y:y+h,x:x+w]
		roi_color = img[y:y+h,x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
    


pd = pandas.DataFrame(listoflists)
pd.to_csv("war0.csv")
cv2.destroyAllWindows()