from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np

# Initialize Flask app
bafapp = Flask(__name__)

# Initialize MediaPipe Pose module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize pose model globally
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angle between three points (shoulder, elbow, and hip)
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A (shoulder)
    b = np.array(b)  # Point B (elbow)
    c = np.array(c)  # Point C (hip)
    
    # Vectors
    ab = b - a
    bc = c - b
    
    # Compute the dot product and magnitudes
    dot = np.dot(ab, bc)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)
    
    # Compute the angle in radians
    angle = np.arccos(dot / (magnitude_ab * magnitude_bc))
    angle = np.degrees(angle)  # Convert to degrees
    
    return angle

# Home route to render the index page (if using HTML frontend)
@bafapp.route('/')
def home():
    return "Welcome to the Posture Detection App!"  # You can replace this with render_template('index.html')

# Endpoint to process video frames (for posture detection)
@bafapp.route('/process_frame', methods=['POST'])
def process_frame():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the file and convert it to a format OpenCV can work with
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"error": "Invalid image file"}), 400
    
    # Recolor image to RGB for pose detection
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    results = pose.process(image_rgb)

    try:
        # Extract landmarks from pose detection
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for the left arm (shoulder, elbow, and hip)
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])

        # Get coordinates for the right arm (shoulder, elbow, and hip)
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        
        # Calculate the angles for both arms
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_hip)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_hip)

        # Posture correction logic
        shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_alignment > 0.02:
            posture_text = "Posture: Incorrect"
        else:
            posture_text = "Posture: Correct"

        return jsonify({
            "left_arm_angle": left_arm_angle,
            "right_arm_angle": right_arm_angle,
            "posture_text": posture_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    bafapp.run(debug=True)
