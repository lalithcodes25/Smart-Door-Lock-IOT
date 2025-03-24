import requests
import face_recognition
import numpy as np
import paho.mqtt.client as mqtt
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from datetime import datetime
import cv2
import json
import time
import os
import pickle
import RPi.GPIO as GPIO

# Initialize ImageKit with your credentials
imagekit = ImageKit(
    private_key='private_p2FXeRi2yj8Kfa5FGqArVA48+js=',
    public_key='public_ldAq2m9B1XL41ABGy0Wj7dBiCmk=',
    url_endpoint='https://ik.imagekit.io/vegnxtdec/img2'
)

# Load known faces (Pre-stored)
known_face_encodings = []
known_face_names = ["Person1"]

known_images = ["face1.jpg"]
for img_path in known_images:
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
    known_face_encodings.append(encoding)

THINGSBOARD_HOST = "mqtt.thingsboard.cloud"
ACCESS_TOKEN = "y1gix1lvjo1qjm9p4j4j"
TOPIC = "v1/devices/me/telemetry" 
RPC_TOPIC = "v1/devices/me/rpc/request/+"

def gate(status):
    if status == 0:
        servo1.ChangeDutyCycle(2 + (0 / 18))
        time.sleep(0.4)
        servo1.ChangeDutyCycle(0)
    else:
        servo1.ChangeDutyCycle(2 + (180 / 18))
        time.sleep(0.5)
        servo1.ChangeDutyCycle(0)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to ThingsBoard MQTT broker")
        client.subscribe(RPC_TOPIC)  # Subscribe to receive instructions
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        method = payload.get("method", "")
        params = payload.get("params", {})

        if method == "open_door" and params.get("state") == "unlock":
            print("Door Unlock Command Received! Triggering unlock mechanism...")
            data = {"status": "granted", "image_url": image_url}
            send_to_thingsboard(data)
            gate(1)
            time.sleep(5)
            gate(0)
        elif method == "open_door" and params.get("state") == "lock":
            print("Door Lock Command Received! Triggering lock mechanism...")
            gate(0)
        else:
            print(f"Unknown command received: {payload}")
    except json.JSONDecodeError:
        print("Failed to decode the received message.")

# Initialize MQTT client
client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.on_connect = on_connect
client.on_message = on_message
client.connect(THINGSBOARD_HOST, 1883, 60)
client.loop_start()

def load_known_faces():
    """Loads known face encodings from stored images."""
    path = "ImageBase"
    known_faces = []
    known_names = []
    
    if not os.path.exists(path):
        os.makedirs(path)

    image_files = os.listdir(path)
    for file in image_files:
        image_path = os.path.join(path, file)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
        known_faces.append(encoding)
        known_names.append(os.path.splitext(file)[0])  # Store name without extension

    return known_faces, known_names

def send_to_thingsboard(data):
    payload = json.dumps(data)
    client.publish(TOPIC, payload)
    print("Data sent to ThingsBoard via MQTT.")

def upload_to_imagekit(file_name):
    """Uploads an image to ImageKit and returns the image URL."""
    with open(file_name, 'rb') as image_file:
        upload_response = imagekit.upload(
            file=image_file,
            file_name=file_name,
            options=UploadFileRequestOptions(
                response_fields=["is_private_file", "tags"],
                tags=["visitor"]
            )
        ).response_metadata.raw

    if upload_response and 'url' in upload_response:
        image_url = upload_response['url']
        print(f"Image uploaded successfully: {image_url}")
        return image_url
    else:
        print("Failed to upload image.")
        return None

# GPIO Setup for Servo Motor (Gate Control)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(22, GPIO.OUT)
servo1 = GPIO.PWM(22, 50)  # Pin 22, PWM frequency 50Hz
servo1.start(0)

print("Loading known faces...")
known_face_encodings, known_face_names = load_known_faces()
print("Face encoding complete.")

# Start video capture
cap = cv2.VideoCapture(0)
time.sleep(2)  # Allow camera to adjust
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Smart Door is active. Waiting for visitors...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    cv2.imshow("Smart Door Camera", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if len(faces) > 0:
        filename = "captured_face.jpg"
        cv2.imwrite(filename, frame)
        print("Face detected! Capturing and processing...")

        unknown_image = face_recognition.load_image_file(filename)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if unknown_encodings:
            unknown_encoding = unknown_encodings[0]
            matches = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)

            if True in matches:
                match_index = np.argmin(face_distances)
                recognized_name = known_face_names[match_index]
                print(f"Access Granted: {recognized_name}")
                gate(1)
                time.sleep(5)
                gate(0)
                data = {"status": "granted", "person_name": recognized_name}
                send_to_thingsboard(data)
            else:
                print("Unknown Face Detected. Uploading image...")
                image_url = upload_to_imagekit(filename)
                if image_url:
                    send_to_thingsboard({"status": "unknown", "image_url": image_url})
                    time.sleep(10)
                    cap.release()
                    cv2.destroyAllWindows()
                    break

cap.release()
cv2.destroyAllWindows()
servo1.stop()
GPIO.cleanup()
print("Smart Door system shut down.")
