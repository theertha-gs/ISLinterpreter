# ISL Interpreter

![Project Banner](https://img.shields.io/badge/ISL%20Interpreter-v1.0-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEwLjUgNy41YzAgLjgyLS42OCAxLjUtMS41IDEuNVM3LjUgOC4zMiA3LjUgNy41IDguMTggNiA5IDZzMS41LjY4IDEuNSAxeiBtMy43MyAxMy41bC0xLjQ0LTEuNDFjLS4zOS0uMzktMS4wMy0uMzktMS40MSAwbC0xLjU1IDEuNTVjLS40NC40NC0xLjE2LjQ0LTEuNiAwbC0xLjU1LTEuNTUtMS40MSAxLjQxYy4zOS4zOSAxLjAyLjM5IDEuNDEgMGwxLjU1LTEuNTVjLjQ0LS40NCAxLjE2LS40NCAxLjYgMGwxLjU1IDEuNTUgMS40MS0xLjQxek0yMCAySDQuQzIuOSAyIDIgMi45IDIgNHYxNmMwIDEuMS45IDIgMiAyaDE2YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bTAgMThINFY0aDE2djE2eiIvPjwvc3ZnPg==)

A full-stack web application that bridges the communication gap by translating Indian Sign Language gestures into text in real-time, featuring live subtitle sharing and custom gesture training.

[![React](https://img.shields.io/badge/React-blue?style=flat&logo=react)](https://react.dev/)
[![Next.js](https://img.shields.io/badge/Next.js-black?style=flat&logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-blue?style=flat&logo=typescript)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-blue?style=flat&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-blue?style=flat&logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-orange?style=flat&logo=google)](https://developers.google.com/mediapipe)

## Key Features

- **Real-Time Gesture Recognition**: Utilizes a webcam feed to interpret ISL gestures and translate them into text with high accuracy and low latency.
- **Live Subtitle Sharing**: Broadcast your translation to others in real-time. Viewers can see the recognized gestures as they are signed via a unique link.
- **User Authentication**: Secure user login and registration system built with **Firebase Authentication**.
- **Custom Gesture Training**: A dedicated interface to record, name, and save new gestures, allowing the system to be expanded and personalized.
- **Advanced Machine Learning Pipeline**:
  - **Feature Extraction**: Uses **MediaPipe** to extract 21 3D hand landmarks from each frame.
  - **Data Augmentation**: Includes scripts to generate synthetic landmark data to improve model robustness.
  - **Efficient Model**: Employs a trained **Random Forest Classifier** for fast and efficient real-time predictions.
- **Modern & Responsive UI**: A sleek user interface built with **Next.js**, **TypeScript**, and **shadcn/ui**, ensuring a seamless experience across devices.
- **WebSocket Communication**: Employs **FastAPI WebSockets** for persistent, bidirectional communication between the client and the server.

##  System Architecture

The application is built on a decoupled frontend-backend architecture, enabling scalability and maintainability.
```
+-------------------------------------+                            +-------------------------------------+
|          Frontend (Next.js)         |    (WebSocket: JSON frames)    |           Backend (FastAPI)         |
+-------------------------------------+ <--------------------------> +-------------------------------------+
| - Camera Feed (WebRTC)            |                            | - WebSocket Connection Manager      |
| - UI Components (shadcn/ui)       |   (Predictions, Status)    | - Frame Processing (OpenCV)         |
| - WebSocket Client                | <-------------------------- | - Hand Landmark Detection (MediaPipe) |
| - User Authentication (Firebase)  |                            | - Gesture Prediction (ML Model)     |
| - Custom Gesture Mgmt             |      (REST API: Auth)      | - Session Management (Broadcast)    |
+-------------------------------------+ -------------------------> +-------------------------------------+
                                            (REST API: Custom Gestures)
```
## Tech Stack

| Category      | Technologies                                                                          |
|---------------|---------------------------------------------------------------------------------------|
| **Frontend**  | `React 19`, `Next.js 15`, `TypeScript`, `Tailwind CSS`, `shadcn/ui`, `Firebase Auth`    |
| **Backend**   | `Python`, `FastAPI`, `WebSockets`, `OpenCV`, `MediaPipe`, `scikit-learn`, `uvicorn`      |
| **Database**  | `Firebase` (for users), `JSON file` (for custom gesture metadata)                       |
| **DevOps**    | `Git`, `npm`, `pip` (Virtual Environment)                                             |

##  Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or later)
- [Python](https://www.python.org/downloads/) (v3.9 or later)
- `git` installed on your machine

1. Clone the Repository

```bash
git clone https://github.com/your-username/theertha-gs-islinterpreter.git
cd theertha-gs-islinterpreter
```
2. Backend Setup
The backend handles the core machine learning and real-time communication.
```bash
# Navigate to the backend directory
cd backend

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the required Python packages
pip install fastapi "uvicorn[standard]" opencv-python mediapipe scikit-learn joblib numpy

# Train the model (this will process data and create model files)
# The `run_pipeline.py` script automates this process.
python run_pipeline.py --train
```

3. Frontend Setup
The frontend is a Next.js application for the user interface.
```Bash
# Navigate to the frontend directory from the root
cd frontend

# Install npm packages
npm install
```

Next, create a file named frontend/.env.local and add your Firebase project configuration:
```Bash
Generated env
NEXT_PUBLIC_FIREBASE_API_KEY="YOUR_API_KEY"
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN="YOUR_AUTH_DOMAIN"
NEXT_PUBLIC_FIREBASE_PROJECT_ID="YOUR_PROJECT_ID"
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET="YOUR_STORAGE_BUCKET"
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID="YOUR_SENDER_ID"
NEXT_PUBLIC_FIREBASE_APP_ID="YOUR_APP_ID"
Env
```
4. Running the Application
You need to run both the backend and frontend servers simultaneously in separate terminals.
```bash
Terminal 1: Start the Backend Server
cd backend
source venv/bin/activate  # Make sure the venv is active
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

The backend server will be running at http://localhost:8000.

Terminal 2: Start the Frontend Server
cd frontend
npm run dev

The frontend application will be available at http://localhost:3000.
```


📂 Project Structure
The project is organized into two main directories: backend and frontend.
```bash

└── theertha-gs-islinterpreter/
    ├── backend/
    │   ├── app.py              # FastAPI server, WebSocket logic, prediction endpoint
    │   ├── train_isl_model.py  # Main script to run the training pipeline
    │   ├── image_to_csv.py     # Extracts hand landmarks from images
    │   ├── csv_to_model.py     # Trains the ML model from landmark data
    │   ├── model/              # Stores the trained model (.pkl), scaler, and labels
    │   └── training_data/      # Directory for storing training/testing images
    │
    └── frontend/
        ├── src/
        │   ├── app/            # Next.js App Router: contains all pages and layouts
        │   │   ├── camera/     # Main page for real-time translation
        │   │   ├── custom-gestures/ # Page to manage custom gestures
        │   │   ├── share/      # Page for viewing a shared live session
        │   │   └── login/      # User authentication pages
        │   ├── lib/            # Firebase config and Auth Context
        │   └── components/     # Reusable React components (shadcn/ui)
        └── public/
            └── custom-gestures/ # Stores saved custom gesture images
```
# 🧠 How The Model Works
The gesture recognition pipeline is a multi-step process designed for real-time performance.
- **Data Collection:** The model is trained on a dataset of ISL gestures. The project includes tools for:
  - Processing a standard image dataset (extract_dataset.py).
  - Generating synthetic hand landmarks to augment the dataset (create_synthetic_landmarks.py).
  - Capturing and labeling new data in real-time (realtime_training.py).
- **Feature Extraction:** For each image or video frame, Google's MediaPipe Hands library is used to detect the hand and extract the 3D coordinates (x, y, z) of 21 key landmarks. This converts the visual information into a numerical feature vector.
- **Model Training:** The extracted landmark data is flattened and standardized. A Random Forest Classifier is trained on this data. This model was chosen for its balance of high accuracy and low prediction latency, making it ideal for real-time applications.
- **Real-Time Inference:**
  - The frontend captures webcam frames and sends them to the backend via a WebSocket.
  - The FastAPI server receives the frame, uses OpenCV and MediaPipe to extract landmarks.
  - The pre-trained scaler and Random Forest model predict the gesture.
  - The prediction and confidence score are sent back to the frontend through the WebSocket to be displayed on the UI.
