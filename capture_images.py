import cv2
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def capture_images(session_id, duration_minutes, output_folder="captured_images"):
    """
    Capture images from the camera at regular intervals based on the session duration.
    
    Parameters:
        session_id (str): Unique session identifier.
        duration_minutes (int): Total duration of the session in minutes.
        output_folder (str): Folder to save captured images.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output folder '{output_folder}' is ready.")
        
        # Calculate the number of images and intervals
        total_images = duration_minutes // 5
        interval_seconds = (duration_minutes * 60) // total_images
        logger.info(f"Session duration: {duration_minutes} minutes, capturing {total_images} images at {interval_seconds} second intervals.")
        
        # Open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Failed to open the camera.")
        
        for i in range(total_images):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to capture image {i + 1}.")
                continue
            
            # Save the captured image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"{session_id}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            logger.info(f"Captured image {i + 1}/{total_images}: {filename}")
            
            time.sleep(interval_seconds)
        
        cap.release()
        logger.info("Image capture completed.")
    except Exception as e:
        logger.error(f"Error during image capture: {e}")

if __name__ == "__main__":
    session_id = "session_001"  # Replace with the actual session ID
    duration_minutes = 40  # Replace with the actual session duration in minutes
    capture_images(session_id, duration_minutes)
