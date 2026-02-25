REAL TIME HAND LANDMARK DETECTION USING ASL

This project aims to develop a hand gesture recognition system using Python, OpenCV, Mediapipe, and the YOLOv5 object detection model. The goal is to classify hand gestures into different classes, such as 'A', 'B', and 'L' in real time, based on the hand landmark positions detected in the camera feed. Additionally, the system can detect nearby objects using the YOLO model to handle cases where the hand gesture may be occluded.

OBJECTIVES:
1.	Object Detection: Use algorithm (eg. YOLO) to detect objects in video frames, identifying multiple object categories such as pedestrians, vehicles, and bicycles in real-time.
2.	Motion Analysis and Tracking: Implement tracking techniques to follow the detected objects as they move through the video, using motion analysis to track their positions across frames.
3.	Real-Time Processing: Ensure the system is capable of processing live video feeds in real-time, balancing accuracy and speed.
4.	Data Visualization: Visualize detected objects and their movement trajectories on the video feed.
5.	Performance Evaluation: Evaluate the performance of the object detection and tracking system by measuring accuracy, frame rate (FPS), and robustness in different environments (e.g., varying lighting or crowded scenes).

WORKFLOW OVERVIEW:
The project begins with data collection and preparation, where users create a customizable dataset of hand gestures. This dataset is used to train a custom hand gesture recognition model, which is integrated with the YOLOv5 object detection model. The system processes real-time video feed to detect hand landmarks and recognize gestures, while also identifying objects in the frame. Visualizations of detected gestures and object trajectories are overlaid on the video, and performance metrics are logged for monitoring. 

TECHNOLOGIES USED: 
1.	Python: The primary programming language used for developing the machine learning models and the overall application.
2.	OpenCV: A library for real-time computer vision tasks, enabling video capture and image processing for gesture recognition.
3.	PyTorch: A deep learning framework utilized for building and training the custom hand gesture recognition model and for integrating the YOLOv5 model.
4.	Ultralytics YOLOv5: An advanced object detection model used to identify and classify multiple objects in the video feed, enhancing the contextual understanding of gestures.
5.	NumPy: A library for numerical computations, used for data manipulation and processing during model training and inference.
6.	Mediapipe: It is used for building multimodal applied machine learning pipelines.
7.	Pickle: It is used for saving trained machine learning models

PROJECT STRUCTURE:
 


CODE

collect_img.py
 
 




create_dataset.py
 
 



train_classifier.py
 



test_classifier.py
 
 
 
 
 


















OUTPUT:
1.	Running collect_img.py
 
 
Created  a dataset:
 

2.	Running create_dataset.py
 
A pickle file of the dataset is created
 

3.	Running train_classifier.py
 

             A pickle file of the model is created
 

4.	Running test_classifier.py
 

The model correctly classifies the sign as letter ‘A’
 
The system can detect multiple objects and detect the sign language at the same time
 
 
 

Real time performance metrics along with object detected logged in console:
 

STRENGTHS OF THE SYSTEM [FEATURES]: 
1.	 Scalable and Customizable Dataset: The system allows users to create and expand a custom dataset tailored to specific hand gestures and environments, enhancing the model's adaptability and performance in diverse scenarios.

2.	Hand Gesture Recognition: Utilizing a custom-built machine learning model, the system accurately implements motion tracking and interprets hand gestures, facilitating real-time sign language translation tailored to user needs.

3.	Data Visualization: The tool visualizes detected objects and their movement trajectories directly on the video feed, allowing users to track interactions and gestures dynamically, enhancing understanding and analysis.

4.	Multi-Object Detection Capabilities: Integrated with the YOLOv5 model, the tool can simultaneously detect and classify multiple objects in the camera frame, providing context that enhances the interpretation of hand gestures.

5.	Real-Time Performance Metrics: The application provides real-time console logs detailing performance metrics such as processing times and detected objects, enabling users to monitor efficiency and responsiveness during operation.

LIMITATIONS:
1.	Variability in Hand Shapes: Differences in hand size, shape, and orientation can impact the accuracy of gesture recognition. Users with different physical attributes may not be recognized as effectively.

2.	Limited Gesture Vocabulary: The system's accuracy is dependent on the dataset used for training. If the dataset does not include a wide variety of gestures or sign languages, the model may struggle to recognize fewer common signs.

3.	User Dependency: The effectiveness of the system may vary between users based on their signing style or familiarity with the gesture vocabulary.

4.	Environmental Factors: Factors such as camera angle, distance from the camera, and movement speed can affect detection accuracy.

