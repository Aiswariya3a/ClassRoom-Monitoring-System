import pandas as pd
import numpy as np

def find_common_viewpoint(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Initialize dictionaries to store median values for each zone
    zones = ["left", "center", "right"]
    zone_median_pose = {}

    # Calculate median head pose for each zone to handle outliers
    for zone in zones:
        zone_data = data[data['zone'] == zone]
        if not zone_data.empty:
            median_pitch = zone_data['pose.pitch'].median()
            median_yaw = zone_data['pose.yaw'].median()
            median_roll = zone_data['pose.roll'].median()
            zone_median_pose[zone] = {
                "median_pitch": median_pitch,
                "median_yaw": median_yaw,
                "median_roll": median_roll
            }
    
    return zone_median_pose

def calculate_engagement(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Define emotion weights based on classroom engagement
    emotion_weights = {
        "neutral": 20,    # Baseline - focused
        "happy": -5,      # Positive engagement
        "sad": 20,       # Disengagement potential
        "angry": 5,     # High disengagement
        "surprise": -10,  # Distraction potential
        "fear": -5,      # Discomfort, disengagement
        "disgust": -30    # Likely disengagement
    }
    
    # Get the zone-wise common viewpoints (pitch, yaw, roll)
    zone_median_pose = find_common_viewpoint(csv_file)
    
    # Initialize list to store the engagement scores
    engagement_scores = []
    
    for _, row in data.iterrows():
        face_id = row["face_id"]
        zone = row["zone"]
        emotion = row["emotion"]
        confidence = row["confidence"]
        emotion_weight = emotion_weights.get(emotion, 0) * confidence  # Weight adjusted by confidence
        
        # Get the common viewpoint for the student's zone
        zone_pose = zone_median_pose.get(zone, {"median_pitch": 0, "median_yaw": 0})
        
        # Calculate deviation from the zone viewpoint
        pitch_deviation = abs(row["pose.pitch"] - zone_pose["median_pitch"])
        yaw_deviation = abs(row["pose.yaw"] - zone_pose["median_yaw"])
        
        # Boundaries for extreme deviations
        if yaw_deviation > 90:
            yaw_deviation = 100  # Assign extreme penalty for large yaw deviation
        if pitch_deviation > 100:
            pitch_deviation = 100  # Assign extreme penalty for large pitch deviation
        
        # Normalize the head pose deviations (dynamically calculated max deviation)
        max_deviation = 45
        yaw_score = max(0, 100 - (yaw_deviation / max_deviation) * 100)
        pitch_score = max(0, 100 - (pitch_deviation / max_deviation) * 100)
        
        # Aggregate head pose scores with yaw as the priority
        head_pose_score = (yaw_score * 0.7) + (pitch_score * 0.3)
        
        # Normalize emotion score to 0-100
        normalized_emotion = (emotion_weight + 50)
        normalized_emotion = np.clip(normalized_emotion, 0, 100)
        
        # Combine the normalized scores for final engagement score
        total_engagement_score = (head_pose_score * 0.8) + (normalized_emotion * 0.2)
        
        # Final normalization of engagement score between 0 and 100
        total_engagement_score = np.clip(total_engagement_score, 0, 100)
        
        # Store the engagement score and relevant data
        engagement_scores.append({
            "face_id": face_id,
            "zone": zone,
            "emotion": emotion,
            "confidence": confidence,
            "emotion_weight": emotion_weight,
            "pitch_deviation": pitch_deviation,
            "yaw_deviation": yaw_deviation,
            "pitch_score": pitch_score,
            "yaw_score": yaw_score,
            "head_pose_score": head_pose_score,
            "normalized_emotion": normalized_emotion,
            "engagement_score": total_engagement_score
        })
    
    # Convert engagement scores into a DataFrame
    engagement_df = pd.DataFrame(engagement_scores)
    
    # Calculate overall class engagement score (average of individual engagement scores)
    overall_engagement_score = engagement_df["engagement_score"].mean()

    # Return both individual scores and overall class engagement score
    return engagement_df, overall_engagement_score

# Example usage
csv_file = "output/face_data.csv"
engagement_df, overall_engagement_score = calculate_engagement(csv_file)

# Print the final engagement scores with emotion and head pose data
print("\nEngagement Scores with Emotion and Head Pose Data:")
print(engagement_df[['face_id', 'zone', 'emotion', 'confidence', 'emotion_weight', 'pitch_deviation', 
                     'yaw_deviation', 'pitch_score', 'yaw_score', 'head_pose_score', 'normalized_emotion', 'engagement_score']])

# Print the overall engagement score for the class
print(f"\nOverall Engagement Score for the Class: {overall_engagement_score:.2f}")


# import pandas as pd
# import numpy as np
# from db_manager import DatabaseManager
# from datetime import datetime
# import logging
# from FaceAnalyzer import FaceAnalyzer


# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# class EngagementCalculator:
#     def __init__(self, db_config):
#         """
#         Initialize the EngagementCalculator with database configuration.
#         """
#         self.db_manager = DatabaseManager(**db_config)

#     def fetch_session_data(self, session_id):
#         """
#         Fetch face data for the given session ID from the database.
#         """
#         query = """
#         SELECT face_id, zone, emotion, confidence, pose_pitch, pose_yaw, pose_roll, pose_confidence, created_at
#         FROM face_data
#         WHERE session_id = %s
#         """
#         data = self.db_manager.fetch_data(query, (session_id,))
#         return pd.DataFrame(data, columns=[
#             "face_id", "zone", "emotion", "confidence", 
#             "pose.pitch", "pose.yaw", "pose.roll", "pose.confidence", "created_at"
#         ])

#     def find_common_viewpoint(self, data):
#         """
#         Calculate the average head pose for each zone (left, center, right).
#         """
#         zones = ["left", "center", "right"]
#         zone_avg_pose = {}

#         for zone in zones:
#             zone_data = data[data['zone'] == zone]
#             if not zone_data.empty:
#                 avg_pitch = zone_data["pose.pitch"].mean()
#                 avg_yaw = zone_data["pose.yaw"].mean()
#                 avg_roll = zone_data["pose.roll"].mean()
#                 zone_avg_pose[zone] = {
#                     "avg_pitch": avg_pitch,
#                     "avg_yaw": avg_yaw,
#                     "avg_roll": avg_roll
#                 }
#         return zone_avg_pose

#     def calculate_engagement(self, session_id):
#         """
#         Calculate engagement scores for the session.
#         """
#         data = self.fetch_session_data(session_id)
#         if data.empty:
#             logger.warning(f"No data found for session ID: {session_id}")
#             return None

#         # Emotion weight mapping
#         emotion_weights = {
#             "neutral": 10,    # Baseline for focused behavior
#             "happy": 5,       # Slightly positive engagement
#             "sad": -5,        # Potential disengagement
#             "angry": -10,     # Likely disengagement
#             "surprise": -5,   # Distraction potential
#             "fear": -10,      # Negative engagement
#             "disgust": -5     # Rare but disengaging
#         }

#         # Calculate common head pose per zone
#         zone_avg_pose = self.find_common_viewpoint(data)

#         # Compute engagement scores
#         engagement_scores = []
#         for _, row in data.iterrows():
#             face_id = row["face_id"]
#             zone = row["zone"]
#             emotion = row["emotion"]
#             weighted_emotion_score = emotion_weights.get(emotion, 0)

#             # Zone-specific pose deviation
#             zone_pose = zone_avg_pose.get(zone, {"avg_pitch": 0, "avg_yaw": 0, "avg_roll": 0})
#             pitch_deviation = abs(row["pose.pitch"] - zone_pose["avg_pitch"])
#             yaw_deviation = abs(row["pose.yaw"] - zone_pose["avg_yaw"])

#             # Normalize head pose deviations (0-100 scale)
#             max_deviation = 45  # Maximum tolerance for distraction
#             pitch_score = max(0, 100 - (pitch_deviation / max_deviation) * 100)
#             yaw_score = max(0, 100 - (yaw_deviation / max_deviation) * 100)

#             # Head pose score and engagement score
#             head_pose_score = (pitch_score + yaw_score) / 2
#             total_engagement_score = (head_pose_score * 0.7) + (weighted_emotion_score * 0.3)
#             total_engagement_score = np.clip(total_engagement_score, 0, 100)  # Ensure within 0-100

#             engagement_scores.append({
#                 "face_id": face_id,
#                 "zone": zone,
#                 "emotion": emotion,
#                 "weighted_emotion_score": weighted_emotion_score,
#                 "pitch_deviation": pitch_deviation,
#                 "yaw_deviation": yaw_deviation,
#                 "head_pose_score": head_pose_score,
#                 "engagement_score": total_engagement_score
#             })

#         # Convert to DataFrame
#         engagement_df = pd.DataFrame(engagement_scores)

#         # Calculate average engagement score for the session
#         avg_engagement_score = engagement_df["engagement_score"].mean()
#         logger.info(f"Average engagement score for session {session_id}: {avg_engagement_score:.2f}")

#         # Save session engagement score to the database
#         self.db_manager.save_engagement_score(session_id, avg_engagement_score)

#         return engagement_df, avg_engagement_score

#     def close(self):
#         """
#         Close the database connection when done.
#         """
#         self.db_manager.close()
#         logger.info("Database connection closed.")

# # Example usage
# if __name__ == "__main__":
#     db_config = {
#         "host": "172.16.30.21",
#         "user": "ai_model",
#         "password": "password",
#         "database": "cms"
#     }
#     session_id = "session_001"  # Replace with the actual session ID

#     # Step 1: Process all captured images
#     folder_path = "captured_images"
#     analyzer = FaceAnalyzer("captured_images",db_config)  # Pass None initially; images are set dynamically
#     analyzer.analyze_images_in_folder(folder_path, session_id)

#     # Step 2: Calculate session engagement score
#     calculator = EngagementCalculator(db_config)
#     engagement_df, avg_engagement_score = calculator.calculate_engagement(session_id)

#     if engagement_df is not None:
#         print("\nEngagement Scores for the Session:")
#         print(engagement_df)
#         print(f"\nAverage Engagement Score for Session {session_id}: {avg_engagement_score:.2f}")



