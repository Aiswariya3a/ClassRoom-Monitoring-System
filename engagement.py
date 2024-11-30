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
