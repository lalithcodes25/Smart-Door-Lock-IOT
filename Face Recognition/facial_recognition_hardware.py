import face_recognition
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from datetime import datetime
import json
import time
import pickle
from gpiozero import AngularServo

# Load pre-trained face encodings
print("[INFO] loading encodings...\n")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the USB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv_scaler = 4  # Scaling factor for performance optimization

face_locations = []
face_encodings = []
face_names = []
thingsboard_response_received = False
thingsboard_response_send = False
frame_count = 0
start_time = time.time()
fps = 0

# List of authorized names
authorized_names = ["sid","yashu"]  # Case-sensitive

imagekit = ImageKit(
    private_key='private_p2FXeRi2yj8Kfa5FGqArVA48+js=',
    public_key='public_ldAq2m9B1XL41ABGy0Wj7dBiCmk=',
    url_endpoint='https://ik.imagekit.io/vegnxtdec/img2'
)

THINGSBOARD_HOST = "mqtt.thingsboard.cloud"
ACCESS_TOKEN = "UkDsP3FOdCaYseM9zma3"
TOPIC = "v1/devices/me/telemetry" 
RPC_TOPIC = "v1/devices/me/rpc/request/+"

def gate(status):
    if status == 0:
        servo.angle = 0
    else:
        servo.angle = 90
    time.sleep(5)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to ThingsBoard MQTT broker")
        client.subscribe(RPC_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        method = payload.get("method", "")
        params = payload.get("params", {})

        if method == "open_door" and params.get("state") == "unlock":
            print("Door Unlock Command Received! Triggering unlock mechanism...")
            send_to_thingsboard({"status": "granted"})
            gate(1)
            time.sleep(5)
            gate(0)
        elif method == "close_door" and params.get("state") == "lock":
            print("Door Lock Command Received! Triggering lock mechanism...")
            gate(0)
        else:
            print(f"Unknown command received: {payload}")
        thingsboard_response_received = True
    except json.JSONDecodeError:
        print("Failed to decode the received message.")


client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.on_connect = on_connect
client.on_message = on_message
client.connect(THINGSBOARD_HOST, 1883, 60)
client.loop_start()

def send_to_thingsboard(data):
    client.publish(TOPIC, json.dumps(data))
    print("Data sent to ThingsBoard via MQTT.")


def upload_to_imagekit(file_name):
    with open(file_name, 'rb') as image_file:
        upload_response = imagekit.upload(
            file=image_file,
            file_name=file_name,
            options=UploadFileRequestOptions(response_fields=["is_private_file", "tags"], tags=["visitor"])
        ).response_metadata.raw
    
    return upload_response.get('url', None)

servo = AngularServo(14, min_angle=0, max_angle=180, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

def process_frame(frame):
    global face_locations, face_encodings, face_names, thingsboard_response_received
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name in authorized_names:
                authorized_face_detected = True
        face_names.append(name)
    
    if authorized_face_detected:
        print(f"Access Granted: {name}")
        send_to_thingsboard({"status": "granted", "person_name": name})
        gate(1)
        time.sleep(5)
        gate(0)
    else:
        filename = "unknown.jpg"
        cv2.imwrite(filename, frame)
        print("Unknown Face Detected. Uploading image...")
        image_url = upload_to_imagekit(filename)
        if image_url:
            send_to_thingsboard({"status": "unk", "image_url": image_url})
            thingsboard_response_send = True
            time.sleep(10)

    return frame

def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
    return frame


while True:
    if not thingsboard_response_received and thingsboard_response_send:
        continue
    thingsboard_response_received = False
    thingsboard_response_send = False
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break
    
    # Process the frame
    processed_frame = process_frame(frame)
    display_frame = draw_results(processed_frame)

    # Display output
    cv2.imshow('Video', display_frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
