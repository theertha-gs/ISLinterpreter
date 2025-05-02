"""
This script creates synthetic hand landmarks data directly for all sign language classes.
This bypasses the MediaPipe hand detection step, which may fail on placeholder images.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import time

# Define classes
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Number of samples to create for each class
num_train_samples = 200  # Increased for more training data
num_test_samples = 50    # Increased for better testing

# MediaPipe hands has 21 landmarks with 3 coordinates (x, y, z) each
num_landmarks = 21
num_coords_per_landmark = 3
total_features = num_landmarks * num_coords_per_landmark

# Create CSV directories
os.makedirs("csv", exist_ok=True)

# Hand landmark indices for reference
# WRIST = 0
# THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4
# INDEX_FINGER_MCP = 5, INDEX_FINGER_PIP = 6, INDEX_FINGER_DIP = 7, INDEX_FINGER_TIP = 8
# MIDDLE_FINGER_MCP = 9, MIDDLE_FINGER_PIP = 10, MIDDLE_FINGER_DIP = 11, MIDDLE_FINGER_TIP = 12
# RING_FINGER_MCP = 13, RING_FINGER_PIP = 14, RING_FINGER_DIP = 15, RING_FINGER_TIP = 16
# PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20

def generate_base_landmark_for_number(number):
    """Generate a base landmark configuration for a number 1-9"""
    # Create a basic hand shape
    base_landmark = np.zeros((num_landmarks, num_coords_per_landmark))
    
    # Set wrist position
    base_landmark[0] = [0.5, 0.8, 0]
    
    # Palm landmarks (fixed for all numbers)
    for i in range(1, 5):  # Thumb base joints
        base_landmark[i] = [0.35 + (i * 0.03), 0.7 - (i * 0.05), 0.05 * i]
    
    # Base positions for fingers
    for finger in range(4):  # 4 fingers excluding thumb
        mcp_idx = 5 + (finger * 4)  # MCP joint of each finger
        base_landmark[mcp_idx] = [0.4 + (finger * 0.06), 0.6, 0.05]
    
    # Adjust finger positions based on the number
    if number == '1':
        # Index finger up, others down
        for i in range(6, 9):  # Index finger extended
            base_landmark[i] = [base_landmark[5][0], 0.6 - ((i - 5) * 0.15), 0.05]
        # Other fingers curled
        for finger in range(1, 4):
            for joint in range(1, 4):
                idx = 5 + (finger * 4) + joint
                base_landmark[idx] = [base_landmark[5 + finger * 4][0] + (joint * 0.02), 
                                      base_landmark[5 + finger * 4][1] + (joint * 0.02), 
                                      0.05]
    
    elif number == '2':
        # Index and middle fingers up
        for finger in range(2):
            for joint in range(1, 4):
                idx = 5 + (finger * 4) + joint
                base_landmark[idx] = [base_landmark[5 + finger * 4][0], 
                                     0.6 - (joint * 0.15), 
                                     0.05]
        # Other fingers curled
        for finger in range(2, 4):
            for joint in range(1, 4):
                idx = 5 + (finger * 4) + joint
                base_landmark[idx] = [base_landmark[5 + finger * 4][0] + (joint * 0.02), 
                                      base_landmark[5 + finger * 4][1] + (joint * 0.02), 
                                      0.05]
    
    elif number in ['3', '4', '5']:
        # Multiple fingers up based on the number
        num_fingers_up = int(number)
        for finger in range(min(num_fingers_up, 4)):
            for joint in range(1, 4):
                idx = 5 + (finger * 4) + joint
                base_landmark[idx] = [base_landmark[5 + finger * 4][0], 
                                     0.6 - (joint * 0.15), 
                                     0.05]
        
        # Curl any remaining fingers
        for finger in range(min(num_fingers_up, 4), 4):
            for joint in range(1, 4):
                idx = 5 + (finger * 4) + joint
                base_landmark[idx] = [base_landmark[5 + finger * 4][0] + (joint * 0.02), 
                                      base_landmark[5 + finger * 4][1] + (joint * 0.02), 
                                      0.05]
                                      
        # For 5, we also extend the thumb
        if number == '5':
            for i in range(2, 5):
                base_landmark[i] = [0.35 - ((i-1) * 0.05), 0.7 - ((i-1) * 0.1), 0.05]
                
    elif number in ['6', '7', '8', '9']:
        # These are more complex hand shapes - we'll simplify
        # For 6-9, we'll use different finger combinations
        
        # Base positions - all fingers slightly bent
        for finger in range(4):
            for joint in range(1, 4):
                idx = 5 + (finger * 4) + joint
                bend_factor = 0.1 if joint == 1 else (0.15 if joint == 2 else 0.12)
                base_landmark[idx] = [base_landmark[5 + finger * 4][0], 
                                     base_landmark[5 + finger * 4][1] - (joint * bend_factor), 
                                     0.05 + (joint * 0.01)]
        
        # Thumb position varies
        thumb_bend = 0.3 if number in ['6', '8'] else 0.1
        for i in range(2, 5):
            base_landmark[i] = [0.35 - ((i-1) * 0.03), 0.7 - ((i-1) * thumb_bend), 0.05]
            
        # Specific finger adjustments for each number
        if number == '6':
            # Pinky extended more
            for joint in range(1, 4):
                idx = 17 + joint  # Pinky joints
                base_landmark[idx] = [base_landmark[17][0], 
                                     0.6 - (joint * 0.15), 
                                     0.05]
        elif number == '7':
            # Index and pinky extended
            for finger in [0, 3]:  # Index and pinky
                for joint in range(1, 4):
                    idx = 5 + (finger * 4) + joint
                    base_landmark[idx] = [base_landmark[5 + finger * 4][0], 
                                         0.6 - (joint * 0.15), 
                                         0.05]
        elif number == '8':
            # Middle and ring fingers more bent
            for finger in [1, 2]:  # Middle and ring
                for joint in range(1, 4):
                    idx = 5 + (finger * 4) + joint
                    base_landmark[idx] = [base_landmark[5 + finger * 4][0] + (joint * 0.03), 
                                         base_landmark[5 + finger * 4][1] - (joint * 0.05), 
                                         0.05]
        elif number == '9':
            # Index finger curved differently
            for joint in range(1, 4):
                idx = 5 + joint  # Index finger joints
                curve_x = 0.02 if joint == 1 else (0.04 if joint == 2 else 0.03)
                base_landmark[idx] = [base_landmark[5][0] + (joint * curve_x), 
                                     0.6 - (joint * 0.12), 
                                     0.05 + (joint * 0.02)]
    
    return base_landmark

def generate_base_landmark_for_letter(letter):
    """Generate a base landmark configuration for a letter A-Z"""
    # Create a basic hand shape
    base_landmark = np.zeros((num_landmarks, num_coords_per_landmark))
    
    # Set wrist position
    base_landmark[0] = [0.5, 0.8, 0]
    
    # Palm landmarks (fixed for all letters)
    for i in range(1, 5):  # Thumb base joints
        base_landmark[i] = [0.35 + (i * 0.03), 0.7 - (i * 0.05), 0.05 * i]
    
    # Base positions for fingers
    for finger in range(4):  # 4 fingers excluding thumb
        mcp_idx = 5 + (finger * 4)  # MCP joint of each finger
        base_landmark[mcp_idx] = [0.4 + (finger * 0.06), 0.6, 0.05]
    
    # Helper to set a finger to straight position
    def set_finger_straight(finger_idx):
        base = 5 + (finger_idx * 4)  # MCP joint
        for joint in range(1, 4):
            idx = base + joint
            base_landmark[idx] = [base_landmark[base][0], 
                                 0.6 - (joint * 0.15), 
                                 0.05]
    
    # Helper to set a finger to curled position
    def set_finger_curled(finger_idx):
        base = 5 + (finger_idx * 4)  # MCP joint
        for joint in range(1, 4):
            idx = base + joint
            base_landmark[idx] = [base_landmark[base][0] + (joint * 0.03), 
                                  base_landmark[base][1] + (joint * 0.02), 
                                  0.05]
    
    # Specific letter configurations
    if letter == 'A':
        # Fist with thumb sticking out to side
        for finger in range(4):
            set_finger_curled(finger)
        for i in range(2, 5):  # Extend thumb to side
            base_landmark[i] = [0.35 - ((i-1) * 0.05), 0.7 - ((i-1) * 0.02), 0.05]
    
    elif letter == 'B':
        # All fingers straight up, thumb tucked
        for finger in range(4):
            set_finger_straight(finger)
        for i in range(2, 5):  # Tuck thumb
            base_landmark[i] = [0.35 + (i * 0.01), 0.7 - (i * 0.02), 0.05]
    
    elif letter == 'C':
        # Curved hand like holding a C
        for finger in range(4):
            base = 5 + (finger * 4)  # MCP joint
            for joint in range(1, 4):
                idx = base + joint
                angle = finger * 0.2 + joint * 0.1
                radius = 0.3 - joint * 0.02
                base_landmark[idx] = [0.5 + radius * np.cos(angle), 
                                     0.5 + radius * np.sin(angle), 
                                     0.05 + joint * 0.01]
        # Adjust thumb to match C shape
        for i in range(2, 5):
            angle = -0.5 - (i-2) * 0.2
            radius = 0.25
            base_landmark[i] = [0.5 + radius * np.cos(angle), 
                               0.5 + radius * np.sin(angle), 
                               0.05]
    
    elif letter == 'D':
        # Index finger straight up, others curled
        set_finger_straight(0)  # Index finger straight
        for finger in range(1, 4):
            set_finger_curled(finger)
        # Thumb touches middle finger
        base_landmark[4] = [base_landmark[10][0], base_landmark[10][1], 0.05]
    
    elif letter == 'E':
        # All fingers curled, together
        for finger in range(4):
            base = 5 + (finger * 4)  # MCP joint
            for joint in range(1, 4):
                idx = base + joint
                base_landmark[idx] = [base_landmark[base][0] + (joint * 0.01), 
                                      base_landmark[base][1] + (joint * 0.03), 
                                      0.05]
        # Thumb folded against palm
        base_landmark[4] = [0.45, 0.65, 0.02]
    
    elif letter == 'F':
        # Index, middle, ring, pinky touch thumb
        for finger in range(4):
            if finger == 0:  # Index finger
                # Special position for F sign
                base_landmark[6] = [0.45, 0.5, 0.05]
                base_landmark[7] = [0.4, 0.52, 0.05]
                base_landmark[8] = [0.35, 0.55, 0.05]
            else:
                set_finger_straight(finger)
        # Thumb touches index finger
        base_landmark[4] = [base_landmark[8][0], base_landmark[8][1], 0.05]
    
    # Continue with more letter configurations...
    # This is just a sample for a few letters
    # You would need to add specific configurations for G-Z
    
    # For letters we haven't specifically configured, create distinctive variations
    # based on the letter's position in the alphabet
    if letter not in 'ABCDEF':
        # Get letter's position (0-25)
        letter_idx = ord(letter) - ord('A')
        
        # Use the letter index to create a unique hand shape
        # We'll vary the extension and angle of fingers
        for finger in range(4):
            # Determine if this finger should be extended based on letter
            # Use bit operations on letter index for consistent patterns
            if (letter_idx & (1 << finger)) > 0:
                # Extended finger with variable angle
                angle = (letter_idx % 8) * 0.1 - 0.4
                for joint in range(1, 4):
                    idx = 5 + (finger * 4) + joint
                    length = joint * 0.13
                    base_landmark[idx] = [
                        base_landmark[5 + finger * 4][0] + np.sin(angle) * length,
                        base_landmark[5 + finger * 4][1] - np.cos(angle) * length,
                        0.05 + joint * 0.01
                    ]
            else:
                # Curled finger with variations
                curl_factor = (letter_idx % 5) * 0.01 + 0.02
                for joint in range(1, 4):
                    idx = 5 + (finger * 4) + joint
                    base_landmark[idx] = [
                        base_landmark[5 + finger * 4][0] + joint * curl_factor,
                        base_landmark[5 + finger * 4][1] + joint * curl_factor,
                        0.05
                    ]
        
        # Vary thumb position based on letter
        thumb_angle = (letter_idx % 6) * 0.2 - 0.5
        for joint in range(2, 5):
            length = (joint-1) * 0.1
            base_landmark[joint] = [
                0.4 + np.cos(thumb_angle) * length,
                0.7 + np.sin(thumb_angle) * length,
                0.05
            ]
    
    return base_landmark

def generate_base_landmark(class_name):
    """Generate a base landmark for any class (number or letter)"""
    if class_name in '123456789':
        return generate_base_landmark_for_number(class_name)
    else:
        return generate_base_landmark_for_letter(class_name)

def add_variation(base_landmark, variation_amount=0.02):
    """Add random variation to the base landmark."""
    variation = np.random.normal(0, variation_amount, base_landmark.shape)
    return base_landmark + variation

def create_samples(class_name, num_samples, is_train=True):
    """Create synthetic landmark samples for a class."""
    base_landmark = generate_base_landmark(class_name)
    
    samples = []
    for i in range(num_samples):
        # Add some random variation to the base landmark
        varied_landmark = add_variation(base_landmark)
        
        # Flatten the landmarks to a single row
        flattened = varied_landmark.flatten()
        
        # Create a path that would match what image_to_csv.py would create
        subset = "train" if is_train else "test"
        path = f"training_data/{subset}/{class_name}/{class_name}_{i+1}.jpg"
        
        # Add row with path, label, and landmarks
        row = [path, class_name] + flattened.tolist()
        samples.append(row)
    
    return samples

def main():
    print("Creating synthetic hand landmarks for sign language classes...")
    start_time = time.time()
    
    # Column names: path, label, followed by landmarks (x0, y0, z0, x1, y1, z1, ...)
    columns = ["image_path", "label"]
    for i in range(num_landmarks):
        for coord in ["x", "y", "z"]:
            columns.append(f"{coord}{i}")
    
    # Create training samples
    print(f"Creating {num_train_samples} training samples for each class...")
    train_samples = []
    for class_name in tqdm(classes, desc="Training data"):
        train_samples.extend(create_samples(class_name, num_train_samples, is_train=True))
    
    train_df = pd.DataFrame(train_samples, columns=columns)
    train_df.to_csv("csv/train_landmarks.csv", index=False)
    print(f"Saved training data with {len(train_df)} samples to csv/train_landmarks.csv")
    
    # Create test samples
    print(f"Creating {num_test_samples} test samples for each class...")
    test_samples = []
    for class_name in tqdm(classes, desc="Testing data"):
        test_samples.extend(create_samples(class_name, num_test_samples, is_train=False))
    
    test_df = pd.DataFrame(test_samples, columns=columns)
    test_df.to_csv("csv/test_landmarks.csv", index=False)
    print(f"Saved testing data with {len(test_df)} samples to csv/test_landmarks.csv")
    
    # Save the class list
    with open("class_names.json", "w") as f:
        json.dump(classes, f)
    print(f"Updated class_names.json with {len(classes)} classes")
    
    elapsed_time = time.time() - start_time
    print(f"\nDone! Created synthetic landmarks data for all classes in {elapsed_time:.2f} seconds.")
    print(f"Created {len(train_df)} training samples and {len(test_df)} test samples.")
    print("You can now proceed to train the model with: python csv_to_model.py")

if __name__ == "__main__":
    main() 