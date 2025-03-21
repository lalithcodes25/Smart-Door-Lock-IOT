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
            tags=["tag1", "tag2"])
        )
    
    # Get the URL of the uploaded image
    return upload_response

# Example usage
response = capture_and_upload_image()
print(f"response recieved: {response}")