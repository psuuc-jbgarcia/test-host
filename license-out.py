from flask import Flask, render_template, Response, jsonify
import easyocr
import cv2
import os
import re
import numpy as np
import json
import time
from datetime import datetime
from ultralytics import YOLO
import firebase_admin
from firebase.firebase_config import db

app = Flask(__name__)



# Firestore client

# Initialize YOLO model and EasyOCR reader
model = YOLO('license_plate_detector.pt')  # Path to your YOLO model
reader = easyocr.Reader(['en'])
def get_camera_url(camera_purpose):
    cameras_ref = db.collection("cameras")
    query = cameras_ref.where("cameraPurpose", "==", camera_purpose).limit(1)
    results = query.stream()

    for doc in results:
        camera_url = doc.to_dict().get("cameraUrl")
        if camera_url:
            return camera_url
    return None  # Return None if no camera URL is found

# Fetch the camera URL for "time_in"
camera_url = get_camera_url("time_out")
# Define paths
video_path = 'test1.mp4'
output_images_dir = 'detected_plates'
# cap = cv2.VideoCapture(camera_url,cv2.CAP_FFMPEG)  # Change to video path if needed
cap = cv2.VideoCapture(1)

if not os.path.exists(output_images_dir):
    os.makedirs(output_images_dir)

# Thresholds for detecting stable plates
STABILITY_THRESHOLD = 3  # Number of frames for stability
MIN_PLATE_LENGTH = 5  # Minimum length for a valid plate

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def enhance_image(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def resize_for_ocr(img, scale_factor=2):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip().upper().replace("-", " ")

def is_valid_plate(text):
    pattern = r'^[A-Z0-9\s]+$'
    return re.match(pattern, text)

# Function to update departure time in Firestore
def update_plate_departure_time(plate_number, departure_time):
    # Query Firestore to find all documents with the matching plate_number
    plates_ref = db.collection('detected_plates')
    query = plates_ref.where('plate_number', '==', plate_number)
    results = query.stream()

    # Iterate through the results and update the departure time where necessary
    updated_count = 0  # To track how many documents were updated
    for plate_doc in results:
        plate_data = plate_doc.to_dict()
        # Check if the departure_time is None
        if plate_data.get('departure_time') is None:
            plate_ref = plates_ref.document(plate_doc.id)  # Get the document reference
            plate_ref.update({'departure_time': departure_time})
            updated_count += 1
            print(f"Departure time updated for plate {plate_number} in document {plate_doc.id}")

    if updated_count == 0:
        print(f"No records found with departure_time None for plate {plate_number}.")
    else:
        print(f"Updated {updated_count} document(s) for plate {plate_number}.")

# Constants for stability and plate validation
STABILITY_THRESHOLD = 3  # Number of frames for stability
MIN_PLATE_LENGTH = 5     # Minimum length for a valid plate
MAX_RETRIES = 10         # Max retry attempts for reading frames
RETRY_DELAY = 2          # Delay (in seconds) between retries
output_images_dir = "output_images"  # Directory to save images

# Flask route to stream the video feed
def generate_frames():
    frame_count = 0
    stable_plate = None
    stable_count = 0

    while cap.isOpened():
        # Retry logic for capturing frames
        for attempt in range(MAX_RETRIES):
            ret, frame = cap.read()
            if ret:
                break  # Frame captured successfully, exit retry loop
            else:
                print(f"Attempt {attempt + 1} to capture frame failed, retrying...")
                time.sleep(RETRY_DELAY)


        # Process frame every 5 frames
        if frame_count % 5 == 0:
            results = model(frame, conf=0.5)
            current_detected_plates = {}

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    license_plate_crop = frame[y1:y2, x1:x2]

                    if license_plate_crop.size == 0:
                        continue

                    preprocessed_crop = preprocess_image(license_plate_crop)
                    enhanced_crop = enhance_image(preprocessed_crop)
                    resized_crop = resize_for_ocr(enhanced_crop)

                    ocr_results = reader.readtext(resized_crop)

                    if not ocr_results:
                        continue

                    detected_text = ' '.join(result[1] for result in ocr_results).strip()
                    normalized_text = normalize_text(detected_text)

                    # Validate the detected plate
                    if not is_valid_plate(normalized_text) or len(normalized_text) < MIN_PLATE_LENGTH:
                        continue

                    # Track plate stability
                    if stable_plate == normalized_text:
                        stable_count += 1
                    else:
                        stable_plate = normalized_text
                        stable_count = 1  # Reset counter on change

                    # Finalize if stable plate is detected
                    if stable_count >= STABILITY_THRESHOLD:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        # Update Firestore with the departure time
                        update_plate_departure_time(stable_plate, timestamp)

                        # Save the license plate image with timestamp
                        image_path = os.path.join(output_images_dir, f"{stable_plate}_{timestamp.replace(':', '-')}.jpg")
                        cv2.imwrite(image_path, license_plate_crop)

                        # Send updated plate data via event stream
                        yield f"data: {json.dumps({'plate_number': stable_plate, 'departure_time': timestamp})}\n\n"

                    # Draw the rectangle around the detected plate and display text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_color = (0, 255, 0)
                    font_thickness = 2
                    cv2.putText(frame, normalized_text, (x1, y1 - 10), font, font_scale, font_color, font_thickness)

        frame_count += 1

        # Convert the frame to JPEG and yield it for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            print("Failed to encode frame to JPEG. Exiting.")
            break  # Exit if frame cannot be encoded


    cap.release()
POLL_INTERVAL = 5

def get_updated_plates():
    plates_ref = db.collection('detected_plates')
    plates_docs = plates_ref.stream()

    plates = []
    for plate in plates_docs:
        plate_data = plate.to_dict()
        plates.append(plate_data)
    return plates
@app.route('/stream_plates')
def stream_plates():
    def generate():
        while True:
            # Get updated plate data from Firestore
            updated_plates = get_updated_plates()
            plates_json = json.dumps(updated_plates)

            # Send updated data to the client via SSE
            yield f"data: {plates_json}\n\n"
            time.sleep(POLL_INTERVAL)

    return Response(generate(), content_type='text/event-stream')

@app.route('/')
def index():
    return render_template('timeout.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_plates')
def detected_plates():
    # Fetch detected plates from Firestore
    plates_ref = db.collection('detected_plates')
    plates_docs = plates_ref.stream()

    plates = []
    for plate in plates_docs:
        plate_data = plate.to_dict()
        plates.append(plate_data)

    return jsonify(plates)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5002)
