import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pygame

# Set up email parameters
sender_email = "shahananwar39@gmail.com"
receiver_email = "shahantw1234@gmail.com"
password = "vsxijhjljqwjmjcy"

# Define classes
classes = ["fire"]

# Load the object detection model
net = cv2.dnn.readNet("yolov3_custom1_1000.weights", "yolov3_custom.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set up email message
msg = MIMEMultipart()
msg['Subject'] = 'Object Detected!'
msg['From'] = sender_email
msg['To'] = receiver_email

# Set up video capture from default camera
def value():
    val = input("Enter file name or press enter to start webcam : \n")
    if val == "":
        val = 0
    return val
    

cap = cv2.VideoCapture(value())

# Initialize Pygame
pygame.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound("alert.mp3")

# Loop through video frames and detect objects in real-time
while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # Perform object detection
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Loop through the detected objects and send email if object is detected
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                class_ids.append(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    if indexes == 0:
        print("HEY BOSS! FIRE IS DETECTED")
        alarm_sound.play()

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # If objects are detected, send email with image attached
    if len(class_ids) > 0:
        img_encode = cv2.imencode('.jpg', img)[1]
        msg.attach(MIMEImage(img_encode.tobytes(), name='image.jpg'))
        text = "An object has been detected in the image!"
        msg.attach(MIMEText(text))
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, password)
            smtp.sendmail(sender_email, receiver_email, msg.as_string())

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the alarm sound
alarm_sound.stop()

# Release the capture
cap.release()
cv2.destroyAllWindows()
