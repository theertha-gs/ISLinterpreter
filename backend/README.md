# ISL Interpreter Backend

This is the backend component of the ISL Interpreter, which recognizes Indian Sign Language (ISL) gestures in real-time.

## Setup and Installation

1. Ensure you have Python 3.8+ installed
2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset Setup

### Option 1: Use a Dataset ZIP File

If you have a zip file containing your sign language dataset:

1. Use the extract_dataset.py script to automatically organize it:
   ```
   python extract_dataset.py path/to/your/dataset.zip
   ```
   
   This script will:
   - Extract the zip file
   - Organize images into train/test folders
   - Create class_names.json with detected sign classes
   - Clean up temporary files

### Option 2: Manual Setup

1. Create the directory structure for your dataset:
   ```
   training_data/
     train/
       A/
       B/
       ... (other sign classes)
     test/
       A/
       B/
       ... (other sign classes)
   ```

2. Extract your sign language dataset images into the appropriate folders:
   - Place approximately 80% of images for each sign class in the corresponding `training_data/train/` folder
   - Place the remaining 20% in the corresponding `training_data/test/` folder

3. Create or update `class_names.json` with your sign classes:
   ```json
   ["A", "B", "C", ...]
   ```
   Include all the sign classes in your dataset.

## Training the Model

### Option 1: Automated Training Process

1. Run the training script which handles processing and training automatically:
   ```
   python train_isl_model.py
   ```
   
   This script will:
   - Process the images and extract hand landmarks
   - Train a Random Forest model
   - Save the model, scaler, and labels

### Option 2: Step-by-Step Training

If you prefer to run each step manually:

1. Process the images to extract hand landmarks:
   ```
   python image_to_csv.py
   ```

2. Train the model with the extracted landmarks:
   ```
   python csv_to_model.py
   ```

## Running the Application

Once the model is trained, start the application with:
```
python app.py
```

This will start a FastAPI server that:
- Opens a WebSocket endpoint at ws://localhost:8000/ws
- Accepts webcam frames, processes them with MediaPipe
- Returns predictions using the trained model

## API Endpoints

- `GET /`: Basic server health check
- `GET /available_signs`: Returns the list of signs the model can recognize
- `WebSocket /ws`: Main endpoint for real-time sign language recognition

## Folder Structure

- `model/`: Contains the trained model files
  - `sign_language_model.pkl`: The trained Random Forest model
  - `sign_language_scaler.pkl`: Feature scaler
  - `sign_labels.pkl`: List of recognized signs
  - `confusion_matrix.png`: Visualization of model performance
  - `feature_importances.csv`: Feature importance rankings
- `csv/`: Contains extracted landmark data
- `training_data/`: Contains organized images for training
  - `train/`: Training images
  - `test/`: Testing images

## Troubleshooting

- **Missing Labels**: If you encounter issues with loading labels, the application will try multiple sources: `sign_labels.pkl`, `sign_labels.json`, and `class_names.json`.
- **Low Accuracy**: If recognition is poor, you may need more training data or try adjusting the model parameters in `csv_to_model.py`.
- **MediaPipe Issues**: If hand landmarks are not being detected well, try adjusting lighting and ensure your hand is clearly visible against the background.
- **Dataset Structure**: Make sure your dataset follows the expected structure with class folders containing images.

## Contributing

Feel free to contribute to this project by:
- Adding more sign language gestures
- Improving model accuracy
- Enhancing the real-time performance

## License

This project is open-source and available under the MIT License. 
