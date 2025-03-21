import requests
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from datetime import datetime
import cv2

# Initialize ImageKit with your credentials
imagekit = ImageKit(
    private_key='private_p2FXeRi2yj8Kfa5FGqArVA48+js=',
    public_key='public_ldAq2m9B1XL41ABGy0Wj7dBiCmk=',
    url_endpoint='https://ik.imagekit.io/vegnxtdec/img2'
)

THINGSPEAK_WRITE_API_KEY = "YOUR_THINGSPEAK_WRITE_API_KEY"
THINGSPEAK_CHANNEL_URL = "https://api.thingspeak.com/update"

def capture_and_upload_image():
    # Capture the image
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture image")
        return None

    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"visitor_{timestamp}.jpg"

    # Save the image locally (optional)
    cv2.imwrite(file_name, frame)
    cam.release()

    # Upload to ImageKit
    with open(file_name, 'rb') as image_file:
        upload_response = imagekit.upload(
            file=image_file,
            file_name=file_name,
            options=UploadFileRequestOptions(
                response_fields=["is_private_file", "tags"],
                tags=["visitor"]
            )
        )

    # Get the URL of the uploaded image
    if upload_response and 'url' in upload_response:
        image_url = upload_response['url']
        print(f"Image uploaded successfully: {image_url}")
        return image_url
    else:
        print("Failed to upload image.")
        return None


def send_to_thingspeak(image_url):
    payload = {
        "api_key": THINGSPEAK_WRITE_API_KEY,
        "field1": image_url
    }
    
    response = requests.get(THINGSPEAK_CHANNEL_URL, params=payload)
    
    if response.status_code == 200:
        print("Data sent to ThingSpeak successfully.")
    else:
        print(f"Failed to send data to ThingSpeak. Status code: {response.status_code}, Response: {response.text}")


# Example Usage
image_url = capture_and_upload_image()
if image_url:
    send_to_thingspeak(image_url)
