"""This script uses OpenCV’s haarcascade (face and eye cascade) to detect face
and eyes in a video feed which can be inputted through a webcam."""
#Import necessary libraries
import cv2 as cv
import numpy as np
#Load face cascade and hair cascade from haarcascades folder
face_cascade 
cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")
#Capture video from webcam
video_capture = cv.VideoCapture(0)
#Read all frames from webcam
while True:
 ret, frame = video_capture.read()
 frame = cv.flip(frame,1) #Flip so that video feed is not flipped, and appears mirror like.
 Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, 1.3, 5)

 for (x,y,w,h) in faces:
  cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
  roi_gray = gray[y:y+h, x:x+w]
  roi_color = frame[y:y+h, x:x+w]
  eyes = eye_cascade.detectMultiScale(roi_gray)
 for (ex,ey,ew,eh) in eyes:
  cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  cv.imshow('Video', frame)
 if(cv.waitKey(1) & 0Xff == ord('q')):
  break
#Finally when video capture is over, release the video capture and
destroyAllWindows
video_capture.release()
cv.destroyAllWindows()

A.3 face_and_eye_detector_single_image.py
"""This script uses OpenCV’s haarcascade (face and eye cascade) to detect face
and eyes in a given input image."""
#Import necessary libraries
import cv2 as cv
import numpy as np
#Load face cascade and hair cascade from haarcascades folder
face_cascade =
cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")
#Read image in img and convert it to grayscale and store in gray.
#Image is converted to grayscale, as face cascade doesn’t require to operate on
coloured images.
Img = cv.imread('images/test.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#Detect all faces in image.
Faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#Draw a rectangle over the face, and detect eyes in faces
for (x,y,w,h) in faces:

 cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 #ROI is region of interest with area having face inside it.
 Roi_gray = gray[y:y+h, x:x+w]
 roi_color = img[y:y+h, x:x+w]
 #Detect eyes in face
 eyes = eye_cascade.detectMultiScale(roi_gray)
 for (ex,ey,ew,eh) in eyes:
 cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
