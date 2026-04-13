from langchain_google_vertexai import ChatVertexAI
import logging
import os
from dotenv import load_dotenv
# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.env'))

# --- Service Account Key Authentication Setup ---
try:
    service_account_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH", r"c:\Users\navee\Downloads\useful-sunset-480704-f5-d259fcad0788.json")
    
    if not os.path.exists(service_account_key_path):
        raise FileNotFoundError(f"Service account key file not found at: {service_account_key_path}")
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
    logging.info(f"GOOGLE_APPLICATION_CREDENTIALS set to: {service_account_key_path}")

except FileNotFoundError as e:
    logging.error(f"Fatal Error: {e}")
    logging.error("Please ensure the service account JSON key file exists at the specified path in your .env or script.")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred during service account setup: {e}")
    raise

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "useful-sunset-480704-f5")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")


def get_chat_model(**overrides):
    config = {
        "model": "gemini-2.5-pro",
        "temperature": 0.0,
        "project": GCP_PROJECT_ID,
        "location": GOOGLE_LOCATION,
    }
    config.update(overrides)

    return ChatVertexAI(**config)