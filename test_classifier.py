import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
from ultralytics import YOLO  

warnings.filterwarnings("ignore", category=UserWarning)

# Load your hand gesture model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Load YOLO model using ultralytics
yolo_model = YOLO("yolov5s.pt") 

# Load class names from the COCO dataset (or any custom dataset you are using)
with open('coco.names', 'r') as f: 
    class_names = f.read().strip().split('\n')

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    # Convert frame to RGB for hand tracking
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Detect objects with YOLO
    yolo_results = yolo_model(frame)  # Perform detection on the frame

    # Draw YOLO detections on frame
    for detection in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        confidence = detection.conf.item()
        class_id = int(detection.cls.item())

        # Draw bounding box and label if confidence is high enough
        if confidence > 0.5:  
            # Get class name from class_id
            class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmark positions
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize landmark positions
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Predict hand gesture
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw a rectangle and label for gesture
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)

            # Check for nearby objects
            for det in yolo_results[0].boxes:
                x_det, y_det, x_det2, y_det2 = map(int, det.xyxy[0])
                if abs(x1 - x_det) < 50 and abs(y1 - y_det) < 50:
                    print("Sign language could not be detected. Holding an object.")

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
