import os
import firebase_admin
from firebase_admin import credentials, firestore

# Path to your service account key JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_KEY_PATH = os.path.join(BASE_DIR, 'parkwatch-4edfa-firebase-adminsdk-yhdwg-6dd2a76744.json')

def initialize_firebase():
    """
    Initialize Firebase app with the provided service account key.
    """
    if not firebase_admin._apps:
        if not os.path.isfile(SERVICE_ACCOUNT_KEY_PATH):
            raise FileNotFoundError(f"Service account key file not found: {SERVICE_ACCOUNT_KEY_PATH}")
        
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred,{
            'storageBucket':'parkwatch-4edfa.appspot.com'
        })
        print("Firebase initialized successfully.")

def get_firestore_db():
    """
    Returns a Firestore client for database interactions.
    """
    return firestore.client()

# Initialize Firebase when the module is loaded
initialize_firebase()

# Expose the Firestore client
db = get_firestore_db()
