import easyocr
import cv2
import os
import re
import numpy as np
import json
from datetime import datetime
import time
from ultralytics import YOLO
from flask import Flask, Response, render_template, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from firebase.firebase_config import db
import subprocess
# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase app


# Image processing functions
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def enhance_image(img):
    if len(img.shape) == 2:  # Grayscale image
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
def run_plate_scripts():
    try:
    

        # Run the second instance with camera index 1
        subprocess.Popen(['python', 'license-out.py',])
        print("Second instance of license.py is running on camera 1...")
    except Exception as e:
        print(f"Error running license.py: {e}")
# Initialize YOLO model and EasyOCR reader
model = YOLO('license_plate_detector.pt')
reader = easyocr.Reader(['en'])

# Define paths
video_path = 'a.mp4'
output_images_dir = 'detected_plates'

if not os.path.exists(output_images_dir):
    os.makedirs(output_images_dir)

def save_to_firebase(detected_plate):
    # Query for plates that have the same plate number
    existing_plate_ref = db.collection('detected_plates').where('plate_number', '==', detected_plate['plate_number']).get()

    if len(existing_plate_ref) == 0:
        # No entry for this plate, add new record as a first-time entry
        detected_plate['arrival_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detected_plate['departure_time'] = None  # Still not departed
        db.collection('detected_plates').add(detected_plate)
        print(f"New entry added: {detected_plate['plate_number']}")
    else:
        # If plate exists, check if it has departure time
        for doc in existing_plate_ref:
            plate_data = doc.to_dict()

            # If the plate has departure time (it has left), treat as a re-entry
            if plate_data['departure_time']:
                # Only save if there is no existing record with arrival_time and departure_time = None
                detected_plate['arrival_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                detected_plate['departure_time'] = None  # No departure yet

                # Check if there is already an active record (arrival_time exists and departure_time is None)
                active_plate_ref = db.collection('detected_plates').where('plate_number', '==', detected_plate['plate_number']).where('departure_time', '==', None).get()

                if len(active_plate_ref) == 0:  # If no active record exists, save it
                    db.collection('detected_plates').add(detected_plate)
                    print(f"Re-entry added: {detected_plate['plate_number']}")
                else:
                    print(f"Re-entry for plate {detected_plate['plate_number']} already exists.")
                return  # Exit once the re-entry is saved or found

            # If the plate is still in the system (arrival_time exists but no departure_time), do nothing
            print(f"Plate {detected_plate['plate_number']} already in system and hasn't left yet.")




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
camera_url = get_camera_url("time_in")

# Initialize video capture
cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0)


frame_count = 0
stable_plate = None
stable_count = 0

@app.route('/')
def index():
    # Render the main page template
    return render_template('timein.html', detected_plates=[])



def generate_video():
    global frame_count, stable_plate, stable_count
    STABILITY_THRESHOLD = 3  # number of frames for stability
    MIN_PLATE_LENGTH = 5  # Minimum length for a valid plate

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Video feed failed. Exiting.")
            break  # Exit if video feed fails

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

                    if not is_valid_plate(normalized_text) or len(normalized_text) < MIN_PLATE_LENGTH:
                        continue

                    current_detected_plates[normalized_text] = time.time()

                    # Check stability
                    if stable_plate == normalized_text:
                        stable_count += 1
                    else:
                        stable_plate = normalized_text
                        stable_count = 1  # reset counter on change

                    # Only finalize if stable
                    if stable_count >= STABILITY_THRESHOLD:
                        current_time = time.time()

                        # If the plate is not already recorded, treat as new entry
                        detected_plate = {
                            'plate_number': stable_plate,
                            'arrival_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'departure_time': None  # No departure time yet
                        }

                        # Save the detected plate to Firebase
                        save_to_firebase(detected_plate)

                        # # Save cropped image
                        # image_path = os.path.join(output_images_dir, f"{stable_plate}.jpg")
                        # cv2.imwrite(image_path, license_plate_crop)

                    # Display the frame with plate number
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_color = (0, 255, 0)
                    font_thickness = 2
                    text_size, _ = cv2.getTextSize(normalized_text, font, font_scale, font_thickness)
                    text_x = x1
                    text_y = y1 - 10
                    background_top_left = (text_x, text_y - text_size[1] - 10)
                    background_bottom_right = (text_x + text_size[0], text_y + 5)

                    cv2.rectangle(frame, background_top_left, background_bottom_right, (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, normalized_text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Convert frame to JPEG and send it as a response to the frontend
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1

@app.route('/video_feed')
def video_feed():
    # Start the video stream
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define your get_detected_plates function to fetch data from Firestore
def get_detected_plates():
    # Fetch data from Firestore (assuming collection 'detected_plates')
    plates_ref = db.collection('detected_plates')
    plates = plates_ref.stream()
    
    detected_plates = []
    for plate in plates:
        plate_data = plate.to_dict()
        detected_plates.append({
            'plate_number': plate_data.get('plate_number'),
            'arrival_time': plate_data.get('arrival_time')
        })
    detected_plates.sort(key=lambda x: datetime.strptime(x['arrival_time'], '%Y-%m-%d %H:%M:%S'), reverse=True)

    return detected_plates

@app.route('/stream_plates')
def stream_plates():
    def generate():
        while True:
            plates = get_detected_plates()  # Fetch the detected plates from Firestore
            yield f"data: {json.dumps(plates)}\n\n"
            time.sleep(1)  # Sleep to simulate a delay between data updates
    return Response(generate(), content_type='text/event-stream')

if __name__ == "__main__":
    run_plate_scripts()
    app.run(debug=True, use_reloader=False, port=5001)
