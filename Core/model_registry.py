from langchain_google_vertexai import ChatVertexAI
import logging
import os
from dotenv import load_dotenv

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load ENV (only for local) ---
load_dotenv()

# --- GCP Config ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")

if not GCP_PROJECT_ID:
    raise ValueError("❌ GCP_PROJECT_ID is not set")

logging.info(f"✅ Using Project: {GCP_PROJECT_ID}")
logging.info(f"✅ Using Location: {GOOGLE_LOCATION}")

# --- Chat Model ---
def get_chat_model(**overrides):
    config = {
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "project": GCP_PROJECT_ID,
        "location": GOOGLE_LOCATION,
    }
    config.update(overrides)

    return ChatVertexAI(**config)