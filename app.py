from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_socketio import SocketIO, emit
import os
import json
import subprocess
import time
from datetime import datetime
from FaceAnalyzer import FaceAnalyzer
from engagement import calculate_engagement
import cv2
from llm import analyze_classroom

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"
socketio = SocketIO(app, cors_allowed_origins="*")

# File Paths
ENGAGEMENT_SCORES_FILE = "static/engagement_scores.json"
PHOTO_UPLOAD_DIR = "uploads/photos"
FACE_ANALYZER_SCRIPT = "FaceAnalyzer.py"
ENGAGEMENT_SCRIPT = "engagement.py"
last_photo_path = "output/annotated_image_1.jpg"


# Ensure directories exist
os.makedirs(PHOTO_UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Utility Functions
def load_engagement_scores():
    """Load engagement scores from file."""
    if os.path.exists(ENGAGEMENT_SCORES_FILE):
        with open(ENGAGEMENT_SCORES_FILE, "r") as file:
            return json.load(file)
    return []

def capture_photo(photo_number):
    """
    Capture a photo using the camera and save it to the upload folder.
    Replace this function with actual camera code.
    """
    # Initialize the camera (0 is typically the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    # Capture a single frame
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        raise Exception("Failed to capture image")

    # Generate a photo path
    photo_path = os.path.join(PHOTO_UPLOAD_DIR, f"photo_{photo_number}.jpg")

    # Save the captured photo
    cv2.imwrite(photo_path, frame)

    # Release the camera
    cap.release()
    
    print(f"Photo {photo_number} captured and saved to {photo_path}")
    
    return photo_path

def save_engagement_scores(scores):
    """Save engagement scores to file."""
    with open(ENGAGEMENT_SCORES_FILE, "w") as file:
        json.dump(scores, file, indent=4)

def analyze_photo(photo_path):
    """
    Analyze a photo using FaceAnalyzer.py and engagement.py.
    Returns the engagement score for the photo.
    
    """
    try:
        # Step 1: Run FaceAnalyzer.py (ensure FaceAnalyzer script is correct)
        face_analyzer_command = ["python3", FACE_ANALYZER_SCRIPT, photo_path]
        subprocess.run(face_analyzer_command, check=True)

        # Step 2: Run engagement.py (ensure engagement script is correct)
        engagement_command = ["python3", ENGAGEMENT_SCRIPT]
        result = subprocess.run(engagement_command, capture_output=True, text=True, check=True)



        # Step 3: Parse engagement score (Assuming engagement.py returns a single score in stdout)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        return f"Error during photo analysis: {e}"

@app.route("/", methods=["GET", "POST"])
def login():
    """Handle login."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "teacher" and password == "password123":  # Simplified for demonstration
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    """Dashboard showing engagement scores."""
    if "user" not in session:
        return redirect(url_for("login"))
    engagement_scores = load_engagement_scores()
    return render_template("dashboard.html", scores=engagement_scores)

@app.route("/start-session", methods=['GET', "POST"])
def start_session():
    """Start the session and analyze engagement at regular intervals."""
    # Folder to save face_data and results
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    engagement_scores = []
    
    # Process each photo for 5 minutes (capturing a photo every minute)
    for i in range(2):
        # Capture photo at 1-minute intervals
        photo_path = capture_photo(i + 1)
        
        # Initialize the FaceAnalyzer and analyze the captured photo
        face_analyzer = FaceAnalyzer(photo_path)
        num_faces = face_analyzer.analyze_faces()
        
        # Save the results to a CSV (this is the face_data CSV)
        face_data_csv = os.path.join(output_folder, f"face_data_{i + 1}.csv")
        face_analyzer.save_results(output_image_path=f"{output_folder}/annotated_image_{i + 1}.jpg", output_csv_path=face_data_csv)
        
        # Calculate engagement based on the newly saved face_data CSV
        engagement_df, overall_engagement_score = calculate_engagement(face_data_csv)
        
        # Log engagement score
        print(f"Engagement Score for photo {i + 1}: {overall_engagement_score}")
        print(engagement_df)

        engagement_scores.append(overall_engagement_score)
        
        # Emit real-time engagement score to dashboard
        socketio.emit("score_update", {"photo_number": i + 1, "engagement_score": overall_engagement_score})
        
        # Wait for 1 minute before capturing the next photo
        if i < 1:
            print("Waiting for 1 minute before the next capture...")
            time.sleep(60)
    
    # Final session score (average of all engagement scores)
    session_engagement_score = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0

    # Update engagement scores file
    all_scores = load_engagement_scores()
    all_scores.append({"session": f"Session {len(all_scores) + 1}", "score": session_engagement_score})
    save_engagement_scores(all_scores)

    try:
        llm_response = analyze_classroom(photo_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"LLM Analysis failed: {str(e)}"})
    
    # Save LLM report
    report_path = "static/classroom_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(llm_response, f, indent=4)

    return jsonify({"success": True, "score": session_engagement_score})

@app.route("/logout")
def logout():
    """Logout user."""
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    socketio.run(app, debug=True)
