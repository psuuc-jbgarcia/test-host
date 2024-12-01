from flask import Flask, render_template, Response, jsonify
import easyocr
import cv2
import os
import re
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLO model and EasyOCR reader
model = YOLO('license_plate_detector.pt')
reader = easyocr.Reader(['en'])

# Define paths
video_path = 'a.mp4'
output_json_path = 'json_file/detected_plates.json'
output_images_dir = 'detected_plates'

if not os.path.exists(output_images_dir):
    os.makedirs(output_images_dir)

# Load existing plate records if they exist
detected_plates_data = []
if os.path.exists(output_json_path):
    if os.path.getsize(output_json_path) > 0:
        try:
            with open(output_json_path, 'r') as json_file:
                detected_plates_data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON data: {e}")

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

# Flask route to stream the video feed
def generate_frames():
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    stable_plate = None
    stable_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            results = model(frame, conf=0.5)
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

                    if stable_plate == normalized_text:
                        stable_count += 1
                    else:
                        stable_plate = normalized_text
                        stable_count = 1  # reset counter on change

                    if stable_count >= STABILITY_THRESHOLD:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        existing_entry = next(
                            (entry for entry in detected_plates_data 
                             if entry['plate_number'] == stable_plate and entry['departure_time'] is None), 
                            None
                        )

                        if existing_entry:
                            existing_entry['departure_time'] = timestamp

                            image_path = os.path.join(output_images_dir, f"{stable_plate}_{timestamp.replace(':', '-')}.jpg")
                            cv2.imwrite(image_path, license_plate_crop)

                        with open(output_json_path, 'w') as json_file:
                            json.dump(detected_plates_data, json_file, indent=4)

                        # Send updated plate data via event stream
                        yield f"data: {json.dumps(detected_plates_data)}\n\n"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_color = (0, 255, 0)
                    font_thickness = 2
                    cv2.putText(frame, normalized_text, (x1, y1 - 10), font, font_scale, font_color, font_thickness)

        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask route for EventSource stream (SSE)
@app.route('/stream_plates')
def stream_plates():
    return Response(generate_frames(), content_type='text/event-stream')

@app.route('/')
def index():
    return render_template('timeout.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_plates')
def detected_plates():
    # Create a copy of detected_plates_data for display purposes
    display_data = []
    for entry in detected_plates_data:
        # Add a temporary field for display
        entry_copy = entry.copy()
        if entry.get('departure_time') is None:
            entry_copy['departure_time'] = 'Still Parked'
        display_data.append(entry_copy)
    
    # Return the display data to the client
    return jsonify(display_data)

if __name__ == "__main__":
    app.run(debug=True)
