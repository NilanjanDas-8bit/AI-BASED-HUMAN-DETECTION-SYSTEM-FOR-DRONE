import cv2
import numpy as np

x_mouse = 0
y_mouse = 0

def mouse_events(event, x, y, flags, param):
    global x_mouse
    global y_mouse
    x_mouse = x
    y_mouse = y

thermal_camera = cv2.VideoCapture('http://192.168.40.163:8080/video')  # Use camera index 0 (default camera)
if not thermal_camera.isOpened():
    print("Error: Could not open camera. Exiting...")
    exit()

thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
thermal_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ' '))
thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)

cv2.imshow('gray8', np.zeros((120, 160, 3), dtype=np.uint8))  # Initialize with a black frame
cv2.setMouseCallback('gray8', mouse_events)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

while True:
    grabbed, frame = thermal_camera.read()

    if not grabbed:
        print("Failed to grab a frame. Exiting...")
        break

    try:
        temperature_pointer = frame[y_mouse, x_mouse]
        temperature_pointer = (temperature_pointer / 100) * 9 / 5 - 459.67
    except IndexError:
        print("Error: Mouse coordinates outside the frame. Exiting...")
        break

    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
    frame = np.uint8(frame)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)
    faces = face_classifier.detectMultiScale(gray, 1.9, 3)

    cv2.circle(frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
    cv2.putText(frame, "{0:.1f} Fahrenheit".format(temperature_pointer), (x_mouse - 40, y_mouse - 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    upper_body = upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in upper_body:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('gray8', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

thermal_camera.release()
cv2.destroyAllWindows()
