"""
This is the main backend script that processes the webcam feed and sends predictions back to the client via WebSocket.
this uses the MediaPipe Hands model to detect hand landmarks in the webcam feed. It then uses a 
pre-trained Random Forest classifier to predict the sign language gesture based on the landmarks.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import json
import uvicorn
from io import BytesIO
from PIL import Image
import os
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from collections import deque, Counter
import logging
import uuid
from datetime import datetime
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize session management
active_sessions: Dict[str, Set[WebSocket]] = {}
broadcaster_sessions: Dict[str, WebSocket] = {}

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load model and related files
model_dir = "model"
model_path = os.path.join(model_dir, "sign_language_model.pkl")
scaler_path = os.path.join(model_dir, "sign_language_scaler.pkl")
labels_path = os.path.join(model_dir, "sign_labels.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    sign_labels = joblib.load(labels_path)
    logger.info(f"Model loaded successfully with {len(sign_labels)} labels")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    scaler = None
    sign_labels = []

# Create a prediction smoothing buffer
prediction_buffer_size = 5
prediction_buffer = deque(maxlen=prediction_buffer_size)

def smooth_predictions(new_prediction: str) -> str:
    prediction_buffer.append(new_prediction)
    counter = Counter(prediction_buffer)
    return counter.most_common(1)[0][0]

def extract_landmarks(frame: np.ndarray) -> Optional[List[float]]:
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return landmarks
        return None
    except Exception as e:
        logger.error(f"Error extracting landmarks: {e}")
        return None

def process_frame(frame: np.ndarray) -> tuple[str, float, Dict[str, float]]:
    try:
        landmarks = extract_landmarks(frame)
        
        if landmarks:
            scaled_landmarks = scaler.transform([landmarks])
            prediction = model.predict(scaled_landmarks)[0]
            probabilities = model.predict_proba(scaled_landmarks)[0]
            probs_dict = {label: float(prob) for label, prob in zip(sign_labels, probabilities)}
            confidence = float(probabilities[model.classes_.tolist().index(prediction)])
            smoothed_prediction = smooth_predictions(prediction)
            return smoothed_prediction, confidence, probs_dict
        return "No hand detected", 0.0, {}
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return "Error processing frame", 0.0, {}

async def broadcast_prediction(session_id: str, data: dict):
    """Broadcast prediction to all viewers of a session"""
    if session_id in active_sessions:
        for viewer in active_sessions[session_id]:
            try:
                await viewer.send_json(data)
            except:
                pass

@app.websocket("/ws/broadcast/{session_id}")
async def websocket_broadcaster(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"Broadcaster connected for session {session_id}")
    
    if model is None:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close()
        return
    
    try:
        # Register broadcaster
        broadcaster_sessions[session_id] = websocket
        active_sessions[session_id] = set()
        
        while True:
            try:
                data = await websocket.receive_text()
                data = json.loads(data)
                
                if "frame" in data:
                    frame_data = data["frame"]
                    frame_bytes = base64.b64decode(frame_data)
                    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        prediction, confidence, probabilities = process_frame(frame)
                        response = {
                            "prediction": prediction,
                            "confidence": confidence,
                            "probabilities": probabilities,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Send to broadcaster
                        await websocket.send_json(response)
                        
                        # Broadcast to viewers
                        await broadcast_prediction(session_id, response)
                        
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
                continue
                
    except WebSocketDisconnect:
        logger.info(f"Broadcaster disconnected from session {session_id}")
        # Clean up session
        if session_id in broadcaster_sessions:
            del broadcaster_sessions[session_id]
        if session_id in active_sessions:
            viewers = active_sessions[session_id]
            for viewer in viewers:
                try:
                    await viewer.close()
                except:
                    pass
            del active_sessions[session_id]
    
    except Exception as e:
        logger.error(f"Error in broadcaster WebSocket: {e}")

@app.websocket("/ws/view/{session_id}")
async def websocket_viewer(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"Viewer connected to session {session_id}")
    
    if session_id not in broadcaster_sessions:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    try:
        # Add viewer to session
        if session_id not in active_sessions:
            active_sessions[session_id] = set()
        active_sessions[session_id].add(websocket)
        
        try:
            while True:
                # Keep connection alive and wait for disconnection
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info(f"Viewer disconnected from session {session_id}")
            if session_id in active_sessions:
                active_sessions[session_id].remove(websocket)
        
    except Exception as e:
        logger.error(f"Error in viewer WebSocket: {e}")
        if session_id in active_sessions and websocket in active_sessions[session_id]:
            active_sessions[session_id].remove(websocket)

@app.get("/api/create-session")
async def create_session():
    """Create a new session ID for broadcasting"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.get("/")
def read_root():
    """Health check endpoint"""
    model_status = "Loaded" if model is not None else "Not loaded"
    return {
        "status": "Sign Language API is running",
        "model_status": model_status,
        "classes": sign_labels if model is not None else []
    }

if __name__ == "__main__":
    logger.info("Starting Sign Language API")
    logger.info(f"Loaded model with {len(sign_labels)} classes: {sign_labels}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
