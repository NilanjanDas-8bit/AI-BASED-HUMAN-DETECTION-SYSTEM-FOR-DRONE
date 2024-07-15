import cv2
import numpy as np

def thermal_like_effect(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a colormap to the grayscale frame (e.g., 'COLORMAP_JET' for a rainbow colormap)
    thermal_colormap = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
    
    return thermal_colormap

cap = cv2.VideoCapture('http://192.168.193.239:8080/video')

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

upper_body_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)
    faces = face_classifier.detectMultiScale(gray, 1.9, 3)
    
    # Display the resulting frame
    for (x,y,w,h) in humans:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(127,0,255),2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
    upper_body = upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE

     )
    # Draw a rectangle around the upper bodies
    for (x, y, w, h) in upper_body:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         
    thermal_frame = thermal_like_effect(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
