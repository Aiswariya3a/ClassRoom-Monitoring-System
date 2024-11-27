# import mysql.connector
# from datetime import datetime
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class DatabaseManager:
#     def __init__(self, host, user, password, database):
#         """
#         Initialize the DatabaseManager with MySQL connection details.
#         """
#         try:
#             self.connection = mysql.connector.connect(
#                 host=host,
#                 user=user,
#                 password=password,
#                 database=database
#             )
#             self.cursor = self.connection.cursor()
#             logger.info("Database connection established successfully.")
#         except mysql.connector.Error as err:
#             logger.error(f"Error: {err}")
#             self.connection = None
#             self.cursor = None

#     def save_face_data(self, face_data):
#         """
#         Save face analysis data into the MySQL database.
        
#         Parameters:
#         face_data (dict): Dictionary containing face analysis data (face_id, zone, emotion, confidence, created_at).
#         """
#         if self.connection is None:
#             logger.error("Failed to connect to the database.")
#             return
        
#         query = """
#         INSERT INTO face_data (face_id, zone, emotion, confidence, created_at)
#         VALUES (%s, %s, %s, %s, %s)
#         """
#         values = (
#             face_data["face_id"],
#             face_data["zone"],
#             face_data["emotion"],
#             face_data["confidence"],
#             face_data["created_at"]
#         )

#         try:
#             self.cursor.execute(query, values)
#             self.connection.commit()
#             logger.info(f"Face data saved for face_id {face_data['face_id']}.")
#         except mysql.connector.Error as err:
#             logger.error(f"Failed to save face data: {err}")

#     def save_engagement_score(self, session_id, engagement_score):
#         """
#         Save engagement score to the temporary table.
        
#         Parameters:
#         session_id (str): Unique identifier for the session.
#         engagement_score (float): Overall engagement score for the session.
#         """
#         if self.connection is None:
#             logger.error("Failed to connect to the database.")
#             return
        
#         timestamp = datetime.now()
#         query = """
#         INSERT INTO session_engagement (session_id, engagement_score, created_at)
#         VALUES (%s, %s, %s)
#         """
#         values = (session_id, engagement_score, timestamp)

#         try:
#             self.cursor.execute(query, values)
#             self.connection.commit()
#             logger.info(f"Engagement score saved for session {session_id}: {engagement_score}.")
#         except mysql.connector.Error as err:
#             logger.error(f"Failed to save engagement score: {err}")

        

#     def close(self):
#         """
#         Close the database connection.
#         """
#         if self.connection is not None:
#             self.cursor.close()
#             self.connection.close()
#             logger.info("Database connection closed.")


import mysql.connector
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, host, user, password, database):
        """Initialize the DatabaseManager with MySQL connection details."""
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            self.cursor = self.connection.cursor()
            logger.info("Database connection established successfully.")
        except mysql.connector.Error as err:
            logger.error(f"Error: {err}")
            self.connection = None
            self.cursor = None

    def fetch_data(self, query, params=None):
        """Fetch data from the database."""
        if self.connection is None:
            logger.error("Failed to connect to the database.")
            return []

        try:
            if params is None:
                self.cursor.execute(query)
            else:
                self.cursor.execute(query, params)
            data = self.cursor.fetchall()
            return data
        except mysql.connector.Error as err:
            logger.error(f"Failed to fetch data: {err}")
            return []

    def save_face_data(self, face_data):
        """Save face data to the database."""
        if self.connection is None:
            logger.error("Failed to connect to the database.")
            return
        
        query = """
        INSERT INTO face_data (face_id, zone, emotion, confidence, created_at, session_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (
            face_data["face_id"],
            face_data["zone"],
            face_data["emotion"],
            face_data["confidence"],
            face_data["created_at"],
            face_data["session_id"]
        )

        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Face data saved for {face_data['face_id']}.")
        except mysql.connector.Error as err:
            logger.error(f"Failed to save face data: {err}")

    def save_engagement_score(self, session_id, engagement_score):
        """Save engagement score to the temporary table."""
        if self.connection is None:
            logger.error("Failed to connect to the database.")
            return
        
        timestamp = datetime.now()
        query = """
        INSERT INTO session_engagement (session_id, engagement_score, created_at)
        VALUES (%s, %s, %s)
        """
        values = (session_id, engagement_score, timestamp)

        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Engagement score saved for session {session_id}: {engagement_score}.")
        except mysql.connector.Error as err:
            logger.error(f"Failed to save engagement score: {err}")

    def close(self):
        """Close the database connection."""
        if self.connection is not None:
            self.cursor.close()
            self.connection.close()
            logger.info("Database connection closed.")
