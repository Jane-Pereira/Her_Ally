from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points (shoulder, elbow, wrist)
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A (shoulder)
    b = np.array(b)  # Point B (elbow)
    c = np.array(c)  # Point C (wrist)
    
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

# Initialize Video Capture
cap = cv2.VideoCapture(0)

counter = 0
stage = None

# Set a threshold for the "down" position where the angle is considered fully bent
DOWN_POSITION_THRESHOLD = 30   # Arm fully bent (near shoulder)
UP_POSITION_THRESHOLD = 160   # Arm fully extended (straight)

# Set the threshold for shoulder alignment (posture correction)
POSTURE_THRESHOLD = 0.01  # Reduced threshold to detect even smaller tilt

# Generator for video stream
def generate_video_stream():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        global counter, stage
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recolor the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make pose detection
            results = pose.process(image)
            
            # Recolor back to BGR for display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for shoulder, elbow, wrist
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate the angle between shoulder, elbow, and wrist
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Arm curl logic for counting
                if angle > UP_POSITION_THRESHOLD:
                    stage = "up"
                if angle < DOWN_POSITION_THRESHOLD and stage == "up":
                    stage = "down"
                    counter += 1
                    print(f"Count: {counter}")
                
                # Detect shoulder tilt for posture correction
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                
                # Calculate the difference in shoulder heights (y-coordinates)
                shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
                
                # Determine the posture tilt
                if shoulder_alignment > POSTURE_THRESHOLD:
                    if left_shoulder.y < right_shoulder.y:  # Left shoulder is higher than right, right tilt
                        posture_text = "Right Tilt"
                        posture_color = (0, 0, 255)  # Red
                    else:  # Right shoulder is higher than left, left tilt
                        posture_text = "Left Tilt"
                        posture_color = (0, 0, 255)  # Red
                else:
                    posture_text = "Correct Posture"
                    posture_color = (0, 255, 0)  # Green

            except:
                posture_text = "Correct Posture"  # Default to correct posture if no landmarks found
                posture_color = (0, 255, 0)  # Green for correct posture
                pass
            
            # Render text with a larger white background box
            rep_text = f'CURL COUNTER: {counter}'
            stage_text = f'STAGE: {stage}'
            
            # Determine the size of the enlarged text box
            box_width = 300
            box_height = 150  # Made the box taller to fit both posture and count text
            box_x = 10
            box_y = 10
            cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)
            
            # Display "Arm Curl Exercise" in the first line and other info in the following lines
            cv2.putText(image, "Arm Curl Exercise", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, rep_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, stage_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, posture_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)

            # Render pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to start video stream
@app.route('/start_exercise')
def start_exercise():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
