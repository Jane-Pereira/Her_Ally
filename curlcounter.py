import cv2
import mediapipe as mp
import numpy as np

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

cap = cv2.VideoCapture(0)

counter = 0
stage = None

# Set a threshold for the "down" position where the angle is considered fully bent
DOWN_POSITION_THRESHOLD = 30   # Arm fully bent (near shoulder)
UP_POSITION_THRESHOLD = 160   # Arm fully extended (straight)

# Set the threshold for shoulder alignment (posture correction)
POSTURE_THRESHOLD = 0.01  # Reduced threshold to detect even smaller tilt

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
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
            
            # Visualize the angle on the frame
            cv2.putText(image, str(int(angle)), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
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

        # Show the frame
        cv2.imshow("Frame", image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
