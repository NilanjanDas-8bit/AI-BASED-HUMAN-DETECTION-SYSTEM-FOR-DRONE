# import the necessary packages
import cv2
import numpy as np

# create mouse global coordinates
x_mouse = 0
y_mouse = 0                 
 
# mouse events function
def mouse_events(event, x, y, flags, param):
    # mouse movement event
    if event == cv2.EVENT_MOUSEMOVE:
    # update global mouse coordinates
      global x_mouse
      global y_mouse
      x_mouse = x
      y_mouse = y

# set up the thermal camera index (thermal_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) on Windows OS)
thermal_camera = cv2.VideoCapture('http://192.168.193.239:8080/video')
# set up the thermal camera resolution
thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# set up the thermal camera to get the gray16 stream and raw data
thermal_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# set up mouse events and prepare the thermal frame display
grabbed, frame= thermal_camera.read()
cv2.imshow('gray8', frame)
cv2.setMouseCallback('gray8', mouse_events)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

upper_body_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while True:
    # grab the frame from the thermal camera stream
    (grabbed, frame) = thermal_camera.read()
    # calculate temperature
    temperature_pointer = frame[y_mouse, x_mouse]
    # temperature_pointer = (temperature_pointer / 100) - 273.15
    temperature_pointer = (temperature_pointer / 100) * 9 / 5 - 459.67
    # convert the gray16 image into a gray8
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
    frame = np.uint8(frame)
  
    # colorized the gray8 image using OpenCV colormaps
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)
    faces = face_classifier.detectMultiScale(gray, 1.9, 3)
    
    # write pointer
    cv2.circle(frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
    # write temperature
    cv2.putText(frame, "{0:.1f} Fahrenheit".format(temperature_pointer), (x_mouse - 40, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
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
    
    # show the thermal frame
    cv2.imshow('gray8', frame)
    cv2.waitKey(1)
# do a bit of cleanup
thermal_camera.release()
cv2.destroyAllWindows()