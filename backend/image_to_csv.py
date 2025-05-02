"""
This script processes images from the training_data directory, extracts hand landmarks using MediaPipe,
and saves the landmarks to CSV files for training.
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def locate_base_folder():
    """Locate the base folder containing training data"""
    # Check current directory
    if os.path.exists('training_data'):
        base_folder = 'training_data'
    # Check parent directory
    elif os.path.exists('../training_data'):
        base_folder = '../training_data'
    else:
        print("❌ Error: Could not find training_data directory")
        return None
    
    print(f"Found training data in: {os.path.abspath(base_folder)}")
    return base_folder

def setup_folders(base_folder):
    """Setup and check folders"""
train_folder = os.path.join(base_folder, "train")
test_folder = os.path.join(base_folder, "test")
    
    # Ensure train and test folders exist
    if not os.path.exists(train_folder):
        print(f"❌ Error: Train folder not found at {train_folder}")
        return None, None
    
    if not os.path.exists(test_folder):
        print(f"⚠️ Warning: Test folder not found at {test_folder}, creating it")
        os.makedirs(test_folder, exist_ok=True)
    
    # Check if folders contain class directories
    train_classes = [d for d in os.listdir(train_folder) 
                   if os.path.isdir(os.path.join(train_folder, d))]
    
    if not train_classes:
        print(f"❌ Error: No class folders found in {train_folder}")
        return None, None
    
    print(f"Found {len(train_classes)} class folders in train directory")
    
    # Create matching test folders if they don't exist
    for class_name in train_classes:
        test_class_dir = os.path.join(test_folder, class_name)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir, exist_ok=True)
            print(f"Created test directory for class {class_name}")
    
    # Make sure csv directory exists
    os.makedirs("csv", exist_ok=True)
    
    return train_folder, test_folder

def extract_landmarks_from_image(image_path, hands):
    """Extract hand landmarks from an image"""
    try:
                # Read image
                image = cv2.imread(image_path)
        if image is None:
            return None, None
                
        # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
        # Process with MediaPipe
                results = hands.process(image_rgb)
        
        # If hand landmarks detected
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks to list
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Convert to numpy array
            return landmarks, image
        else:
            return None, image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def process_images(folder, subset, hands, total_processed=0, total_landmarks=0, save_debug_images=False):
    """
    Process all images in a folder, extract landmarks, and return a DataFrame
    
    Parameters:
    - folder: The folder containing class subfolders with images
    - subset: Either "train" or "test" for logging
    - hands: MediaPipe hands instance
    - total_processed: Counter for processed images
    - total_landmarks: Counter for successful landmarks
    - save_debug_images: If True, save debug images with landmarks drawn
    
    Returns:
    - pandas DataFrame with landmarks, number of processed images, number of landmarks
    """
    
    # Get all class folders
    class_folders = [f for f in os.listdir(folder) 
                   if os.path.isdir(os.path.join(folder, f))]
    
    if not class_folders:
        print(f"No class folders found in {folder}")
        return None, total_processed, total_landmarks
    
    # Prepare data structure for landmarks
    data = []
    debug_dir = os.path.join("debug_images", subset) if save_debug_images else None
    
    if save_debug_images:
        os.makedirs(debug_dir, exist_ok=True)
    
    # For each class folder
    for class_folder in tqdm(class_folders, desc=f"Processing {subset} classes"):
        class_path = os.path.join(folder, class_folder)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in {class_path}")
            continue
        
        # Create debug subdirectory for this class
        if save_debug_images:
            class_debug_dir = os.path.join(debug_dir, class_folder)
            os.makedirs(class_debug_dir, exist_ok=True)
        
        # Process each image in this class
        class_landmarks = 0
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            total_processed += 1
            
            # Extract landmarks
            landmarks, image = extract_landmarks_from_image(image_path, hands)
            
            if landmarks:
                # Save landmark data
                row = [image_path, class_folder] + landmarks
                data.append(row)
                total_landmarks += 1
                class_landmarks += 1
                
                # Save debug image if requested
                if save_debug_images and image is not None:
                    # Draw landmarks on image
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width, _ = image.shape
                    
                    # Create a debug visualization
                    debug_image = image.copy()
                    
                    # Draw hand skeleton
                    mp_drawing.draw_landmarks(
                        debug_image,
                        mp_hands.HandLandmark(
                            [mp_hands.HandLandmark(i) for i in range(21)]
                        ),
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Save debug image
                    debug_path = os.path.join(class_debug_dir, f"debug_{image_file}")
                    cv2.imwrite(debug_path, debug_image)
            
        # Log results for this class
        if class_landmarks > 0:
            print(f"  Class {class_folder}: {class_landmarks}/{len(image_files)} images with landmarks")
                else:
            print(f"  ⚠️ Class {class_folder}: No landmarks detected in any images")
    
    if not data:
        print(f"No landmarks extracted from {subset} images")
        return None, total_processed, total_landmarks
    
    # Create column names for the DataFrame
    columns = ["image_path", "label"]
    for i in range(21):  # 21 landmarks
        for coord in ["x", "y", "z"]:  # 3 coordinates per landmark
            columns.append(f"{coord}{i}")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df, total_processed, total_landmarks

def save_class_list(class_list):
    """Save the list of classes to a JSON file"""
    with open("class_names.json", "w") as f:
        json.dump(class_list, f)
    print(f"Saved {len(class_list)} classes to class_names.json")

def main():
    print("Starting hand landmark extraction...")
    
    # Locate base folder
    base_folder = locate_base_folder()
    if base_folder is None:
        return
    
    # Setup folders
    train_folder, test_folder = setup_folders(base_folder)
    if train_folder is None or test_folder is None:
        return
    
    # Initialize MediaPipe hands
    print("Initializing MediaPipe Hands...")
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    
    # Process train images
    print("\nProcessing training images...")
    train_df, train_processed, train_landmarks = process_images(
        train_folder, "train", hands, save_debug_images=False
    )
    
    # Process test images
    print("\nProcessing test images...")
    test_df, test_processed, test_landmarks = process_images(
        test_folder, "test", hands, 
        total_processed=train_processed, 
        total_landmarks=train_landmarks,
        save_debug_images=False
    )
    
    # Log processing statistics
    total_processed = train_processed + test_processed
    total_landmarks = train_landmarks + test_landmarks
    
    print("\nProcessing complete:")
    print(f"  Total images processed: {total_processed}")
    print(f"  Total landmarks extracted: {total_landmarks}")
    print(f"  Success rate: {total_landmarks/total_processed*100:.1f}%")
    
    # Save DataFrames to CSV
    if train_df is not None and len(train_df) > 0:
        train_df.to_csv("csv/train_landmarks.csv", index=False)
        print(f"Saved {len(train_df)} training landmarks to csv/train_landmarks.csv")
        
        # Save class list
        unique_classes = sorted(train_df["label"].unique().tolist())
        save_class_list(unique_classes)
    else:
        print("❌ No training landmarks extracted. CSV file not created.")
    
    if test_df is not None and len(test_df) > 0:
        test_df.to_csv("csv/test_landmarks.csv", index=False)
        print(f"Saved {len(test_df)} testing landmarks to csv/test_landmarks.csv")
    else:
        print("❌ No testing landmarks extracted. CSV file not created.")
    
    if total_landmarks == 0:
        print("\n❌ No landmarks were extracted from any images.")
        print("This could be because:")
        print("  - The images don't contain visible hands")
        print("  - The hand positioning makes detection difficult")
        print("  - The image quality is too low for MediaPipe to detect hands")
        print("\nConsider using the synthetic landmarks generator instead:")
        print("  python create_synthetic_landmarks.py")
    else:
        print("\nLandmark extraction successful!")
        print("You can now train the model with:")
        print("  python csv_to_model.py")

if __name__ == "__main__":
    main()