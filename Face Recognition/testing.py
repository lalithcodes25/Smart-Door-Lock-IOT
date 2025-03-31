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
import sys
import traceback
import threading

def load_encodings(filename):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    except Exception as e:
        print(f"Error loading encodings: {e}")
        sys.exit(1)

# Load Haar cascade for face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Haar cascade xml file not loaded properly")
except Exception as e:
    print(f"Error loading Haar cascade: {e}")
    sys.exit(1)

print("[INFO] loading encodings...\n")
known_face_encodings, known_face_names = load_encodings("encodings.pickle")

# Initialize the USB camera
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
except Exception as e:
    print(f"[ERROR] Camera error: {e}")
    sys.exit(1)

cv_scaler = 4  # Scaling factor for performance optimization
face_locations = []
face_encodings = []
face_names = []

# Global flags for synchronization and to ensure single processing per face appearance
thingsboard_response_event = threading.Event()
face_processed = False  # This flag prevents reprocessing the same face repeatedly

# List of authorized names
authorized_names = ["sid"]

# Initialize ImageKit
try:
    imagekit = ImageKit(
        private_key='private_p2FXeRi2yj8Kfa5FGqArVA48+js=',
        public_key='public_ldAq2m9B1XL41ABGy0Wj7dBiCmk=',
        url_endpoint='https://ik.imagekit.io/vegnxtdec/img2'
    )
except Exception as e:
    print(f"Error initializing ImageKit: {e}")
    sys.exit(1)

THINGSBOARD_HOST = "mqtt.thingsboard.cloud"
ACCESS_TOKEN = "UkDsP3FOdCaYseM9zma3"
TOPIC = "v1/devices/me/telemetry"
RPC_TOPIC = "v1/devices/me/rpc/request/+"

# Initialize servo
try:
    servo = AngularServo(14, min_angle=0, max_angle=180, 
                         min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
except Exception as e:
    print(f"Error initializing servo: {e}")
    sys.exit(1)

def gate(status):
    try:
        if status == 0:
            servo.angle = 0
        else:
            servo.angle = 180
        time.sleep(5)
    except Exception as e:
        print(f"Error in gate function: {e}")

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
        thingsboard_response_event.set()  # Mark response as received
    except Exception as e:
        print(f"Error in on_message: {e}")
        traceback.print_exc()

client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(THINGSBOARD_HOST, 1883, 60)
except Exception as e:
    print(f"Error connecting to MQTT broker: {e}")
    sys.exit(1)

client.loop_start()

def send_to_thingsboard(data):
    try:
        thingsboard_response_event.clear()  # Clear event before sending new data
        client.publish(TOPIC, json.dumps(data))
        print("Data sent to ThingsBoard via MQTT.")
    except Exception as e:
        print(f"Error sending data to ThingsBoard: {e}")

def upload_to_imagekit(file_name):
    try:
        with open(file_name, 'rb') as image_file:
            upload_response = imagekit.upload(
                file=image_file,
                file_name=file_name,
                options=UploadFileRequestOptions(response_fields=["is_private_file", "tags"], tags=["visitor"])
            ).response_metadata.raw
        return upload_response.get('url', None)
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None

def process_frame(frame):
    global face_locations, face_encodings, face_names, face_processed
    try:
        # Use Haar cascade to detect human features first.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If no face is detected, reset flag and skip processing.
        if len(faces) == 0:
            face_processed = False
            return frame

        # If a face is already processed, skip further processing.
        if face_processed:
            return frame

        # Process using face_recognition on a scaled frame
        resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_resized_frame)
        face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
        
        face_names = []
        authorized_face_detected = False
        name = "Unknown"
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name in authorized_names:
                    authorized_face_detected = True
            face_names.append(name)
        
        # Set the flag so that this face is not processed repeatedly
        face_processed = True
        
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
                # Wait for ThingsBoard response (with a timeout of 10 seconds)
                if thingsboard_response_event.wait(timeout=10):
                    print("ThingsBoard response received. Proceeding.")
                else:
                    print("Timeout waiting for ThingsBoard response.")
        return frame
    except Exception as e:
        print(f"Error in process_frame: {e}")
        traceback.print_exc()
        return frame

def draw_results(frame):
    try:
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
    except Exception as e:
        print(f"Error in draw_results: {e}")
        return frame

def calculate_fps(frame_count, start_time):
    try:
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        return fps, current_time
    except Exception as e:
        print(f"Error calculating FPS: {e}")
        return 0, time.time()

def main_loop():
    frame_count = 0
    start_time = time.time()
    last_capture_time = time.time()  # Time when the last capture was done
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break
            frame_count += 1
            current_time = time.time()
            # Only process a frame every 5 seconds
            if current_time - last_capture_time >= 2:
                processed_frame = process_frame(frame)
                last_capture_time = current_time
            else:
                processed_frame = frame

            display_frame = draw_results(processed_frame)
            fps, start_time = calculate_fps(frame_count, start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', display_frame)
            if cv2.waitKey(1) == ord("q"):
                break
        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the main loop in a separate thread if needed
    main_thread = threading.Thread(target=main_loop, daemon=True)
    main_thread.start()
    main_thread.join()
