from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
import os
from FaceAnalyzer import FaceAnalyzer
from db_manager import DatabaseManager
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Database credentials
DB_CONFIG = {
    "host": "172.16.30.21",
    "user": "ai_model",
    "password": "password",
    "database": "cms"
}

# Directory to store uploaded images temporarily
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Database Manager
db_manager = DatabaseManager(**DB_CONFIG)

class ProcessedImageResponse(BaseModel):
    face_count: int
    message: str

@app.post("/upload/")
async def upload_images(files: list[UploadFile] = File(...)):
    """
    Upload images for face analysis.

    Parameters:
    - files (list[UploadFile]): List of image files to process.

    Returns:
    - dict: Status and face count for each image.
    """
    try:
        session_id = str(uuid.uuid4())  # Generate unique session ID
        face_count_total = 0

        for file in files:
            # Save the uploaded file temporarily
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Process the file with FaceAnalyzer
            analyzer = FaceAnalyzer(file_path)
            face_count = analyzer.analyze_faces()

            face_count_total += face_count

            # Remove the temporary file after processing
            os.remove(file_path)

        return {"status": "success", "session_id": session_id, "total_faces": face_count_total}

    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/session/{session_id}/process")
async def process_session(session_id: str, background_tasks: BackgroundTasks):
    """
    Process the engagement score for a session.

    Parameters:
    - session_id (str): Unique identifier for the session.
    """
    try:
        # Run engagement score calculation in the background
        background_tasks.add_task(calculate_session_engagement, session_id)
        return {"status": "Processing", "session_id": session_id}

    except Exception as e:
        logger.error(f"Error processing session: {e}")
        return {"status": "error", "message": str(e)}


def calculate_session_engagement(session_id: str):
    """
    Calculate and save the average engagement score for a session.

    Parameters:
    - session_id (str): Unique identifier for the session.
    """
    try:
        # Calculate session engagement score
        avg_score = db_manager.calculate_session_engagement(session_id)
        if avg_score is not None:
            db_manager.save_engagement_score(session_id, avg_score)
            logger.info(f"Session engagement score saved: {avg_score}")
        else:
            logger.error(f"Failed to calculate engagement score for session {session_id}")

    except Exception as e:
        logger.error(f"Error in session engagement calculation: {e}")


@app.get("/session/{session_id}/engagement")
async def get_session_engagement(session_id: str):
    """
    Get the average engagement score for a session.

    Parameters:
    - session_id (str): Unique identifier for the session.

    Returns:
    - dict: Engagement score for the session.
    """
    try:
        avg_score = db_manager.calculate_session_engagement(session_id)
        if avg_score is not None:
            return {"status": "success", "session_id": session_id, "engagement_score": avg_score}
        else:
            return {"status": "error", "message": "Session engagement score not found."}
    except Exception as e:
        logger.error(f"Error retrieving engagement score: {e}")
        return {"status": "error", "message": str(e)}


@app.on_event("shutdown")
def shutdown_event():
    """
    Close the database connection on app shutdown.
    """
    db_manager.close()
    logger.info("Database connection closed on shutdown.")
