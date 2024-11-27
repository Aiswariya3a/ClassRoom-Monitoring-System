# import cv2
# import numpy as np
# import pandas as pd
# import face_detection
# from deepface import DeepFace
# from datetime import datetime
# from scipy.spatial.transform import Rotation
# import logging
# import mysql.connector

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# class FaceAnalyzer:
#     def __init__(self, image_path):
#         """Initialize the FaceAnalyzer with an image path."""
#         self.image_path = image_path
#         self.original_image = self.load_image()
#         self.image_height, self.image_width = self.original_image.shape[:2]
#         self.face_data = []
#         self.zones = self.define_zones()

#     def load_image(self):
#         """Load and validate the input image."""
#         image = cv2.imread(self.image_path)
#         if image is None:
#             raise ValueError(f"Could not load image from {self.image_path}")
        
#         min_size = 224  # Minimum size for reliable face detection
#         if image.shape[0] < min_size or image.shape[1] < min_size:
#             image = cv2.resize(image, (min_size, min_size))
#             logger.warning("Image was resized to minimum dimensions for reliable detection")
#         return image

#     def define_zones(self):
#         """Define zones based on the image width."""
#         third_width = self.image_width // 3
#         return {
#             "left": (0, third_width),
#             "center": (third_width, 2 * third_width),
#             "right": (2 * third_width, self.image_width)
#         }

#     def calculate_head_pose(self, landmarks):
#         """Calculate head pose angles using PnP."""
#         try:
#             model_points = np.array([
#                 (0.0, 0.0, 0.0),          # Nose tip
#                 (0.0, -330.0, -65.0),     # Chin
#                 (-225.0, 170.0, -135.0),  # Left eye corner
#                 (225.0, 170.0, -135.0),   # Right eye corner
#                 (-150.0, -150.0, -125.0), # Left mouth corner
#                 (150.0, -150.0, -125.0)   # Right mouth corner
#             ])

#             image_points = [
#                 landmarks["nose"],
#                 landmarks.get("chin", landmarks["nose"]),
#                 landmarks["left_eye"],
#                 landmarks["right_eye"],
#                 landmarks.get("mouth_left", landmarks["left_eye"]),
#                 landmarks.get("mouth_right", landmarks["right_eye"])
#             ]
#             image_points = np.array(image_points, dtype="double")

#             focal_length = self.image_width
#             center = (self.image_width / 2, self.image_height / 2)
#             camera_matrix = np.array([
#                 [focal_length, 0, center[0]],
#                 [0, focal_length, center[1]],
#                 [0, 0, 1]
#             ], dtype="double")
#             dist_coeffs = np.zeros((4, 1))

#             success, rotation_vector, translation_vector = cv2.solvePnP(
#                 model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
#             )
#             if not success:
#                 raise ValueError("PnP solution failed.")

#             rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#             rotation = Rotation.from_matrix(rotation_matrix)
#             euler_angles = rotation.as_euler("xyz", degrees=True)
#             pitch, yaw, roll = euler_angles

#             return {
#                 "pitch": round(pitch, 2),
#                 "yaw": round(yaw, 2),
#                 "roll": round(roll, 2),
#                 "confidence": 1.0
#             }
#         except Exception as e:
#             logger.warning(f"PnP head pose calculation failed: {str(e)}")
#             return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}

#     def detect_emotion(self, face_image):
#         """Detect emotion with confidence score."""
#         try:
#             analysis = DeepFace.analyze(
#                 face_image, actions=["emotion"], enforce_detection=False, silent=True
#             )
#             emotions = analysis[0]["emotion"]
#             dominant_emotion = max(emotions.items(), key=lambda x: x[1])
#             return {"emotion": dominant_emotion[0], "confidence": round(dominant_emotion[1] / 100, 2)}
#         except Exception as e:
#             logger.error(f"Emotion detection error: {str(e)}")
#             return {"emotion": "unknown", "confidence": 0.0}

#     def determine_zone(self, center_x):
#         """Determine which zone the student is located in based on x-coordinate."""
#         for zone, (start, end) in self.zones.items():
#             if start <= center_x < end:
#                 return zone
#         return "unknown"

#     def analyze_faces(self):
#         """Main method to detect and analyze faces in the image."""
#         try:
#             faces = face_detection.detect_faces(self.original_image)
#             for face_id, face in faces.items():
#                 x1, y1, x2, y2 = face["facial_area"]
#                 face_image = self.original_image[y1:y2, x1:x2]

#                 pose = self.calculate_head_pose(face["landmarks"])
#                 emotion_data = self.detect_emotion(face_image)
#                 zone = self.determine_zone((x1 + x2) // 2)

#                 face_data = {
#                     "face_id": face_id,
#                     "zone": zone,
#                     "pose": pose,
#                     "emotion": emotion_data["emotion"],
#                     "confidence": emotion_data["confidence"],
#                     "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
#                 }

#                 # Save to DB
#                 self.save_face_data(face_data)

#             return len(self.face_data)
        
#         except Exception as e:
#             logger.error(f"Face analysis failed: {str(e)}")
#             return 0
        
#     def save_face_data(self, face_id, zone, emotion, confidence):
#         """Save face analysis data into the database."""
#         face_data = {
#             "face_id": face_id,
#             "zone": zone,
#             "emotion": emotion,
#             "confidence": confidence,
#             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
#         }
#         self.db_manager.save_face_data(face_data)
        
#     def draw_face_analysis(self, face_data):
#         """Draw analysis results on the image."""
#         pos = face_data["position"]
#         zone = face_data["zone"]
#         text_lines = [
#             f"Emotion: {face_data['emotion']} ({face_data['confidence']:.2f})",
#             f"Zone: {zone}",
#         ]
#         for i, text in enumerate(text_lines):
#             y = pos["y1"] - 10 - (i * 20)
#             if y > 0:
#                 cv2.putText(self.original_image, text, (pos["x1"], y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     def save_results(self, output_image_path, output_csv_path):
#         """Save the annotated image and CSV."""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_image_path = output_image_path or f"face_analysis_{timestamp}.jpg"
#         output_csv_path = output_csv_path or f"face_data_{timestamp}.csv"

#         # Save annotated image
#         cv2.imwrite(output_image_path, self.original_image)
#         logger.info(f"Annotated image saved to {output_image_path}")

#         # Save face data to CSV
#         df = pd.json_normalize(self.face_data)
#         df.to_csv(output_csv_path, index=False)
#         logger.info(f"Face data saved to {output_csv_path}")

# def main():
#     image_path = "samples/crowd (1).jpeg"  # Replace with your image path
#     output_image = "output/detected_faces_output.jpg"
#     output_csv = "output/face_data.csv"

#     try:
#         analyzer = FaceAnalyzer(image_path)
#         num_faces = analyzer.analyze_faces()

#         if num_faces > 0:
#             analyzer.save_results(output_image, output_csv)
#             logger.info(f"Successfully analyzed {num_faces} faces")
#         else:
#             logger.warning("No faces detected in the image")
#     except Exception as e:
#         logger.error(f"Analysis failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# import pandas as pd
# import face_detection
# from deepface import DeepFace
# from math import atan2, degrees, sqrt
# import logging
# from datetime import datetime
# from inference_sdk import InferenceHTTPClient
# from scipy.spatial.transform import Rotation


# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class FaceAnalyzer:
#     def __init__(self, image_path, activity_client):
#         """Initialize the FaceAnalyzer with an image path and activity detection client."""
#         self.image_path = image_path
#         self.original_image = self.load_image()
#         self.image_height, self.image_width = self.original_image.shape[:2]
#         self.face_data = []
#         self.activity_client = activity_client  # Client for activity detection
        
#     def load_image(self):
#         """Load and validate the input image."""
#         image = cv2.imread(self.image_path)
#         if image is None:
#             raise ValueError(f"Could not load image from {self.image_path}")
        
#         # Validate image size and quality
#         min_size = 224  # minimum size for reliable face detection
#         if image.shape[0] < min_size or image.shape[1] < min_size:
#             image = cv2.resize(image, (min_size, min_size))
#             logger.warning("Image was resized to minimum dimensions for reliable detection")
            
#         return image

#     def calculate_head_pose(self, landmarks):
#         """Calculate head pose angles using PnP with fallback for missing landmarks."""
#         try:
#             # Define 3D model points of facial landmarks in a fixed coordinate system
#             model_points = np.array([
#                 (0.0, 0.0, 0.0),          # Nose tip
#                 (0.0, -330.0, -65.0),     # Chin
#                 (-225.0, 170.0, -135.0),  # Left eye corner
#                 (225.0, 170.0, -135.0),   # Right eye corner
#                 (-150.0, -150.0, -125.0), # Left mouth corner
#                 (150.0, -150.0, -125.0)   # Right mouth corner
#             ])

#             # Map 2D image points from detected landmarks
#             image_points = [
#                 landmarks["nose"],           # Nose tip
#                 landmarks.get("chin", landmarks["nose"]),  # Chin fallback to nose
#                 landmarks["left_eye"],       # Left eye corner
#                 landmarks["right_eye"],      # Right eye corner
#                 landmarks.get("mouth_left", landmarks["left_eye"]),  # Fallback to left eye
#                 landmarks.get("mouth_right", landmarks["right_eye"]) # Fallback to right eye
#             ]

#             image_points = np.array(image_points, dtype="double")

#             # Camera matrix approximation
#             focal_length = self.image_width
#             center = (self.image_width / 2, self.image_height / 2)
#             camera_matrix = np.array([
#                 [focal_length, 0, center[0]],
#                 [0, focal_length, center[1]],
#                 [0, 0, 1]
#             ], dtype="double")

#             # No lens distortion
#             dist_coeffs = np.zeros((4, 1))

#             # Solve PnP to find rotation and translation vectors
#             success, rotation_vector, translation_vector = cv2.solvePnP(
#                 model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
#             )

#             if not success:
#                 raise ValueError("PnP solution failed.")

#             # Convert rotation vector to rotation matrix
#             rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

#             # Calculate Euler angles from the rotation matrix
#             rotation = Rotation.from_matrix(rotation_matrix)
#             euler_angles = rotation.as_euler('xyz', degrees=True)

#             pitch, yaw, roll = euler_angles

#             return {
#                 'pitch': round(pitch, 2),
#                 'yaw': round(yaw, 2),
#                 'roll': round(roll, 2),
#                 'confidence': 1.0  # High confidence since PnP is robust
#             }
#         except Exception as e:
#             logger.warning(f"PnP head pose calculation failed: {str(e)}")
#             return {
#                 'pitch': 0.0,
#                 'yaw': 0.0,
#                 'roll': 0.0,
#                 'confidence': 0.0  # Low confidence for fallback
#             }

#     def calculate_engagement_score(self, emotion, pitch, yaw, roll):
#         """Calculate engagement score based on emotion, pitch, yaw, and roll."""
#         # Start with an engagement score
#         engagement_score = 100

#         # Emotion-based adjustment
#         if emotion in ['angry','surprise']:
#             # Check pitch, yaw, roll for distraction
#                 engagement_score = 80  # Medium engagement if not highly distracted
#         elif emotion in ['neutral']:
#             # High engagement for positive emotions
#             engagement_score = 100  # High engagement

#         # Adjust based on pitch
#         if abs(pitch) < -50 or abs(pitch) > 50:
#             engagement_score -= 10  # Decrease engagement for large pitch deviation

#         # Adjust based on yaw (looking too much to the side is a sign of distraction)
#         if abs(yaw) > 40 or abs(yaw) < 40:
#             engagement_score -= 10  # Low engagement if looking too far sideways

#         # Adjust based on roll (significant head tilting is a sign of distraction)
#         if abs(roll) > 60 or abs(roll) < -200:
#             engagement_score -= 40  # Low engagement for high roll deviation

#         # Normalize the score to 0-100 range
#         engagement_score = max(0, min(100, engagement_score))

#         return engagement_score

#     def detect_emotion(self, face_image):
#         """Detect emotion with confidence score."""
#         try:
#             analysis = DeepFace.analyze(
#                 face_image, 
#                 actions=["emotion"], 
#                 enforce_detection=False,
#                 silent=True
#             )
#             emotions = analysis[0]['emotion']
#             dominant_emotion = max(emotions.items(), key=lambda x: x[1])
#             return {
#                 'emotion': dominant_emotion[0],
#                 'confidence': round(dominant_emotion[1] / 100, 2)
#             }
#         except Exception as e:
#             logger.error(f"Emotion detection error: {str(e)}")
#             return {'emotion': 'unknown', 'confidence': 0.0}

#     def detect_activity(self):
#         """Detect activities in the image using the activity detection model."""
#         try:
#             result = self.activity_client.infer(self.image_path, model_id="student-behavior-detection-v3/2")
#             predictions = result.get("predictions", [])
#             return predictions  # Return raw predictions for postprocessing
#         except Exception as e:
#             logger.error(f"Activity detection failed: {str(e)}")
#             return []

#     def analyze_faces(self):
#         """Main method to detect and analyze faces in the image."""
#         try:
#             faces = face_detection.detect_faces(self.original_image)
#             activities = self.detect_activity()  # Perform activity detection
            
#             for face_id, face in faces.items():
#                 # Extract face area with margin
#                 margin = 0.1
#                 x1, y1, x2, y2 = face["facial_area"]
#                 w, h = x2 - x1, y2 - y1
#                 x1 = max(0, int(x1 - margin * w))
#                 y1 = max(0, int(y1 - margin * h))
#                 x2 = min(self.image_width, int(x2 + margin * w))
#                 y2 = min(self.image_height, int(y2 + margin * h))
                
#                 face_image = self.original_image[y1:y2, x1:x2]
                
#                 # Get face analysis results
#                 pose = self.calculate_head_pose(face["landmarks"])
#                 emotion_data = self.detect_emotion(face_image)
                
#                 # Calculate engagement score
#                 engagement_score = self.calculate_engagement_score(
#                     emotion_data['emotion'], pose['pitch'], pose['yaw'], pose['roll']
#                 )
                
#                 # Find matching activity detection for this face (based on bounding box overlap)
#                 detected_activity = "None"
#                 for activity in activities:
#                     ax1 = activity["x"]
#                     ay1 = activity["y"]
#                     ax2 = ax1 + activity["width"]
#                     ay2 = ay1 + activity["height"]
#                     # Check if activity box overlaps with face box
#                     if x1 <= ax2 and x2 >= ax1 and y1 <= ay2 and y2 >= ay1:
#                         detected_activity = activity["class"]
#                         break
                
#                 # Store face data
#                 face_data = {
#                     "face_id": face_id,
#                     "position": {
#                         "x1": x1, "y1": y1,
#                         "x2": x2, "y2": y2,
#                         "center_x": (x1 + x2) // 2,
#                         "center_y": (y1 + y2) // 2
#                     },
#                     "size": {"width": x2 - x1, "height": y2 - y1},
#                     "pose": pose,
#                     "emotion": emotion_data['emotion'],
#                     "confidence": emotion_data['confidence'],
#                     "activity": detected_activity,
#                     "engagement": engagement_score  # Store the calculated engagement score
#                 }
                
#                 self.face_data.append(face_data)
                
#                 # Draw results on image
#                 self.draw_face_analysis(face_data)
                
#             return len(self.face_data)
            
#         except Exception as e:
#             logger.error(f"Face analysis failed: {str(e)}")
#             return 0

#     def draw_face_analysis(self, face_data):
#         """Draw analysis results on the image."""
#         pos = face_data["position"]
#         activity = face_data.get("activity", "None")
        
#         # Draw face rectangle
#         cv2.rectangle(self.original_image, 
#                      (pos["x1"], pos["y1"]), 
#                      (pos["x2"], pos["y2"]), 
#                      (255, 0, 0), 2)
        
#         # Prepare text
#         text_lines = [
#             f"Emotion: {face_data['emotion']} ({face_data['confidence']:.2f})",
#             f"Activity: {activity}",
#             f"Engagement: {face_data['engagement']:.2f}"
#         ]
        
#         for i, text in enumerate(text_lines):
#             y = pos["y1"] - 10 - (i * 20)
#             if y > 0:
#                 cv2.putText(self.original_image, text,
#                             (pos["x1"], y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     def save_results(self, output_image_path=None, output_csv_path=None):
#         """Save analysis results to image and CSV files."""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         if output_image_path is None:
#             output_image_path = f"face_analysis_{timestamp}.jpg"
#         if output_csv_path is None:
#             output_csv_path = f"face_data_{timestamp}.csv"
            
#         # Save annotated image
#         cv2.imwrite(output_image_path, self.original_image)
#         logger.info(f"Annotated image saved to {output_image_path}")
        
#         # Save face data to CSV
#         df = pd.json_normalize(self.face_data)
#         df.to_csv(output_csv_path, index=False)
#         logger.info(f"Face data saved to {output_csv_path}")


# import cv2
# import numpy as np
# import pandas as pd
# import face_detection
# from deepface import DeepFace
# from datetime import datetime
# from inference_sdk import InferenceHTTPClient
# from scipy.spatial.transform import Rotation
# import logging

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class FaceAnalyzer:
#     def __init__(self, image_path, activity_client):
#         """Initialize the FaceAnalyzer with an image path and activity detection client."""
#         self.image_path = image_path
#         self.original_image = self.load_image()
#         self.image_height, self.image_width = self.original_image.shape[:2]
#         self.face_data = []
#         self.activity_client = activity_client  # Client for activity detection
#         self.zones = self.define_zones()  # Define zones based on image width
        
#     def load_image(self):
#         """Load and validate the input image."""
#         image = cv2.imread(self.image_path)
#         if image is None:
#             raise ValueError(f"Could not load image from {self.image_path}")
        
#         # Validate image size and quality
#         min_size = 224  # minimum size for reliable face detection
#         if image.shape[0] < min_size or image.shape[1] < min_size:
#             image = cv2.resize(image, (min_size, min_size))
#             logger.warning("Image was resized to minimum dimensions for reliable detection")
            
#         return image

#     def define_zones(self):
#         """Define zones based on the image width."""
#         third_width = self.image_width // 3
#         return {
#             "left": (0, third_width),
#             "center": (third_width, 2 * third_width),
#             "right": (2 * third_width, self.image_width)
#         }

#     def calculate_head_pose(self, landmarks):
#         """Calculate head pose angles using PnP with fallback for missing landmarks."""
#         try:
#             # Define 3D model points of facial landmarks in a fixed coordinate system
#             model_points = np.array([
#                 (0.0, 0.0, 0.0),          # Nose tip
#                 (0.0, -330.0, -65.0),     # Chin
#                 (-225.0, 170.0, -135.0),  # Left eye corner
#                 (225.0, 170.0, -135.0),   # Right eye corner
#                 (-150.0, -150.0, -125.0), # Left mouth corner
#                 (150.0, -150.0, -125.0)   # Right mouth corner
#             ])

#             # Map 2D image points from detected landmarks
#             image_points = [
#                 landmarks["nose"],           # Nose tip
#                 landmarks.get("chin", landmarks["nose"]),  # Chin fallback to nose
#                 landmarks["left_eye"],       # Left eye corner
#                 landmarks["right_eye"],      # Right eye corner
#                 landmarks.get("mouth_left", landmarks["left_eye"]),  # Fallback to left eye
#                 landmarks.get("mouth_right", landmarks["right_eye"]) # Fallback to right eye
#             ]

#             image_points = np.array(image_points, dtype="double")

#             # Camera matrix approximation
#             focal_length = self.image_width
#             center = (self.image_width / 2, self.image_height / 2)
#             camera_matrix = np.array([
#                 [focal_length, 0, center[0]],
#                 [0, focal_length, center[1]],
#                 [0, 0, 1]
#             ], dtype="double")

#             # No lens distortion
#             dist_coeffs = np.zeros((4, 1))

#             # Solve PnP to find rotation and translation vectors
#             success, rotation_vector, translation_vector = cv2.solvePnP(
#                 model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
#             )

#             if not success:
#                 raise ValueError("PnP solution failed.")

#             # Convert rotation vector to rotation matrix
#             rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

#             # Calculate Euler angles from the rotation matrix
#             rotation = Rotation.from_matrix(rotation_matrix)
#             euler_angles = rotation.as_euler('xyz', degrees=True)

#             pitch, yaw, roll = euler_angles

#             return {
#                 'pitch': round(pitch, 2),
#                 'yaw': round(yaw, 2),
#                 'roll': round(roll, 2),
#                 'confidence': 1.0  # High confidence since PnP is robust
#             }
#         except Exception as e:
#             logger.warning(f"PnP head pose calculation failed: {str(e)}")
#             return {
#                 'pitch': 0.0,
#                 'yaw': 0.0,
#                 'roll': 0.0,
#                 'confidence': 0.0  # Low confidence for fallback
#             }

#     def detect_emotion(self, face_image):
#         """Detect emotion with confidence score."""
#         try:
#             analysis = DeepFace.analyze(
#                 face_image, 
#                 actions=["emotion"], 
#                 enforce_detection=False,
#                 silent=True
#             )
#             emotions = analysis[0]['emotion']
#             dominant_emotion = max(emotions.items(), key=lambda x: x[1])
#             return {
#                 'emotion': dominant_emotion[0],
#                 'confidence': round(dominant_emotion[1] / 100, 2)
#             }
#         except Exception as e:
#             logger.error(f"Emotion detection error: {str(e)}")
#             return {'emotion': 'unknown', 'confidence': 0.0}

#     def determine_zone(self, center_x):
#         """Determine which zone the student is located in based on x-coordinate."""
#         for zone, (start, end) in self.zones.items():
#             if start <= center_x < end:
#                 return zone
#         return "unknown"
    

#     def analyze_faces(self):
#         """Main method to detect and analyze faces in the image."""
#         try:
#             faces = face_detection.detect_faces(self.original_image)
            
#             for face_id, face in faces.items():
#                 # Extract face area with margin
#                 margin = 0.1
#                 x1, y1, x2, y2 = face["facial_area"]
#                 w, h = x2 - x1, y2 - y1
#                 x1 = max(0, int(x1 - margin * w))
#                 y1 = max(0, int(y1 - margin * h))
#                 x2 = min(self.image_width, int(x2 + margin * w))
#                 y2 = min(self.image_height, int(y2 + margin * h))
                
#                 face_image = self.original_image[y1:y2, x1:x2]
                
#                 # Get face analysis results
#                 pose = self.calculate_head_pose(face["landmarks"])
#                 emotion_data = self.detect_emotion(face_image)
                
#                 # Determine zone based on face position
#                 zone = self.determine_zone((x1 + x2) // 2)
                
#                 # Store face data
#                 face_data = {
#                     "face_id": face_id,
#                     "zone": zone,
#                     "pose": pose,
#                     "emotion": emotion_data['emotion'],
#                     "confidence": emotion_data['confidence'],
#                     "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
#                 }

#                 self.face_data.append(face_data)
                
#                 # Draw results on image
#                 self.draw_face_analysis(face_data)
                
#             return len(self.face_data)
            
#         except Exception as e:
#             logger.error(f"Face analysis failed: {str(e)}")
#             return 0

#     def draw_face_analysis(self, face_data):
#         """Draw analysis results on the image."""
#         pos = face_data["position"]
#         activity = face_data.get("activity", "None")
        
#         # Draw face rectangle
#         cv2.rectangle(self.original_image, 
#                      (pos["x1"], pos["y1"]), 
#                      (pos["x2"], pos["y2"]), 
#                      (255, 0, 0), 2)
        
#         # Prepare text
#         text_lines = [
#             f"Emotion: {face_data['emotion']} ({face_data['confidence']:.2f})",
#             f"Activity: {activity}",
#             f"Engagement: {face_data['engagement']:.2f}"
#         ]
        
#         for i, text in enumerate(text_lines):
#             y = pos["y1"] - 10 - (i * 20)
#             if y > 0:
#                 cv2.putText(self.original_image, text,
#                             (pos["x1"], y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     def save_results(self, output_image_path=None, output_csv_path=None):
#         """Save analysis results to image and CSV files."""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         if output_image_path is None:
#             output_image_path = f"face_analysis_{timestamp}.jpg"
#         if output_csv_path is None:
#             output_csv_path = f"face_data_{timestamp}.csv"
            
#         # Save annotated image
#         cv2.imwrite(output_image_path, self.original_image)
#         logger.info(f"Annotated image saved to {output_image_path}")
        
#         # Save face data to CSV
#         df = pd.json_normalize(self.face_data)
#         df.to_csv(output_csv_path, index=False)
#         logger.info(f"Face data saved to {output_csv_path}")

# def main():
#     """Main execution function."""
#     image_path = "samples/crowd (4).jpeg"  # Replace with your image path
#     output_image = "output/detected_faces_output.jpg"
#     output_csv = "output/face_data.csv"
    
#     # Set up the InferenceHTTPClient
#     activity_client = InferenceHTTPClient(
#         api_url="https://detect.roboflow.com",
#         api_key="Dm7IUK7kb9vb67dEghIg"
#     )
    
#     try:
#         analyzer = FaceAnalyzer(image_path, activity_client)
#         num_faces = analyzer.analyze_faces()
        
#         if num_faces > 0:
#             analyzer.save_results(output_image, output_csv)
#             logger.info(f"Successfully analyzed {num_faces} faces")
#         else:
#             logger.warning("No faces detected in the image")
            
#     except Exception as e:
#         logger.error(f"Analysis failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import pandas as pd
import face_detection
from deepface import DeepFace
from datetime import datetime
from scipy.spatial.transform import Rotation
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FaceAnalyzer:
    def __init__(self, image_path):
        """Initialize the FaceAnalyzer with an image path."""
        self.image_path = image_path
        self.original_image = self.load_image()
        self.image_height, self.image_width = self.original_image.shape[:2]
        self.face_data = []
        self.zones = self.define_zones()

    def load_image(self):
        """Load and validate the input image."""
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        min_size = 224  # Minimum size for reliable face detection
        if image.shape[0] < min_size or image.shape[1] < min_size:
            image = cv2.resize(image, (min_size, min_size))
            logger.warning("Image was resized to minimum dimensions for reliable detection")
        return image

    def define_zones(self):
        """Define zones based on the image width."""
        third_width = self.image_width // 3
        return {
            "left": (0, third_width),
            "center": (third_width, 2 * third_width),
            "right": (2 * third_width, self.image_width)
        }

    def calculate_head_pose(self, landmarks):
        """Calculate head pose angles using PnP."""
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # Left eye corner
                (225.0, 170.0, -135.0),   # Right eye corner
                (-150.0, -150.0, -125.0), # Left mouth corner
                (150.0, -150.0, -125.0)   # Right mouth corner
            ])

            image_points = [
                landmarks["nose"],
                landmarks.get("chin", landmarks["nose"]),
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks.get("mouth_left", landmarks["left_eye"]),
                landmarks.get("mouth_right", landmarks["right_eye"])
            ]
            image_points = np.array(image_points, dtype="double")

            focal_length = self.image_width
            center = (self.image_width / 2, self.image_height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                raise ValueError("PnP solution failed.")

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler("xyz", degrees=True)
            pitch, yaw, roll = euler_angles

            return {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "confidence": 1.0
            }
        except Exception as e:
            logger.warning(f"PnP head pose calculation failed: {str(e)}")
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}

    def detect_emotion(self, face_image):
        """Detect emotion with confidence score."""
        try:
            analysis = DeepFace.analyze(
                face_image, actions=["emotion"], enforce_detection=False, silent=True
            )
            emotions = analysis[0]["emotion"]
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            return {"emotion": dominant_emotion[0], "confidence": round(dominant_emotion[1] / 100, 2)}
        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
            return {"emotion": "unknown", "confidence": 0.0}

    def determine_zone(self, center_x):
        """Determine which zone the student is located in based on x-coordinate."""
        for zone, (start, end) in self.zones.items():
            if start <= center_x < end:
                return zone
        return "unknown"

    def analyze_faces(self):
        """Main method to detect and analyze faces in the image."""
        try:
            faces = face_detection.detect_faces(self.original_image)
            for face_id, face in faces.items():
                x1, y1, x2, y2 = face["facial_area"]
                face_image = self.original_image[y1:y2, x1:x2]

                pose = self.calculate_head_pose(face["landmarks"])
                emotion_data = self.detect_emotion(face_image)
                zone = self.determine_zone((x1 + x2) // 2)

                face_data = {
                    "face_id": face_id,
                    "position": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "center_x": (x1 + x2) // 2,
                        "center_y": (y1 + y2) // 2
                    },
                    "zone": zone,
                    "pose": pose,
                    "emotion": emotion_data["emotion"],
                    "confidence": emotion_data["confidence"],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
                }
                self.face_data.append(face_data)
                self.draw_face_analysis(face_data)

            return len(self.face_data)
        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            return 0

    def draw_face_analysis(self, face_data):
        """Draw analysis results on the image."""
        pos = face_data["position"]
        zone = face_data["zone"]
        text_lines = [
            f"Emotion: {face_data['emotion']} ({face_data['confidence']:.2f})",
            f"Zone: {zone}",
        ]
        for i, text in enumerate(text_lines):
            y = pos["y1"] - 10 - (i * 20)
            if y > 0:
                cv2.putText(self.original_image, text, (pos["x1"], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def save_results(self, output_image_path, output_csv_path):
        """Save the annotated image and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = output_image_path or f"face_analysis_{timestamp}.jpg"
        output_csv_path = output_csv_path or f"face_data_{timestamp}.csv"

        # Save annotated image
        cv2.imwrite(output_image_path, self.original_image)
        logger.info(f"Annotated image saved to {output_image_path}")

        # Save face data to CSV
        df = pd.json_normalize(self.face_data)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Face data saved to {output_csv_path}")

def main():
    image_path = "samples/crowd (1).jpeg"  # Replace with your image path
    output_image = "output/detected_faces_output.jpg"
    output_csv = "output/face_data.csv"

    try:
        analyzer = FaceAnalyzer(image_path)
        num_faces = analyzer.analyze_faces()

        if num_faces > 0:
            analyzer.save_results(output_image, output_csv)
            logger.info(f"Successfully analyzed {num_faces} faces")
        else:
            logger.warning("No faces detected in the image")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()