# firebase_setup.py
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    cred = credentials.Certificate('parkwatch-4edfa-firebase-adminsdk-yhdwg-6dd2a76744.json')  # Replace with your actual path
    firebase_admin.initialize_app(cred)
    return firestore.client()
