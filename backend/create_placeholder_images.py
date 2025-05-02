"""
This script creates placeholder images for sign language classes.
These are just colored squares with the class name, to be used for testing.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# Define classes
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Define paths
train_folder = os.path.join("training_data", "train")
test_folder = os.path.join("training_data", "test")

# Number of images to create for each class
num_train_images = 20
num_test_images = 5

def create_images_for_class(class_name, folder, num_images):
    """Create placeholder images for a class"""
    # Create folder if it doesn't exist
    class_folder = os.path.join(folder, class_name)
    os.makedirs(class_folder, exist_ok=True)
    
    # Generate a random color for this class (for visual distinction)
    color = np.random.randint(0, 256, size=3).tolist()
    
    for i in range(num_images):
        # Create a colored image (300x300 pixels)
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        # Draw a colored rectangle in the center
        cv2.rectangle(img, (50, 50), (250, 250), color, -1)
        
        # Add text with the class name
        cv2.putText(img, class_name, (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
        
        # Add a hand-like shape
        # This is just to make the MediaPipe hand detector more likely to detect something
        cv2.line(img, (150, 250), (150, 150), (0, 0, 0), 5)  # Wrist to palm
        cv2.line(img, (150, 150), (100, 50), (0, 0, 0), 5)   # Thumb
        cv2.line(img, (150, 150), (130, 50), (0, 0, 0), 5)   # Index finger
        cv2.line(img, (150, 150), (150, 50), (0, 0, 0), 5)   # Middle finger
        cv2.line(img, (150, 150), (170, 50), (0, 0, 0), 5)   # Ring finger
        cv2.line(img, (150, 150), (200, 50), (0, 0, 0), 5)   # Pinky
        
        # Add some noise for variety
        noise = np.random.randint(0, 30, size=img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Save the image
        image_path = os.path.join(class_folder, f"{class_name}_{i+1}.jpg")
        cv2.imwrite(image_path, img)

def main():
    print("Creating placeholder images for sign language classes...")
    
    # Create training images
    print(f"Creating {num_train_images} training images for each class...")
    for class_name in tqdm(classes, desc="Training data"):
        create_images_for_class(class_name, train_folder, num_train_images)
    
    # Create test images
    print(f"Creating {num_test_images} test images for each class...")
    for class_name in tqdm(classes, desc="Testing data"):
        create_images_for_class(class_name, test_folder, num_test_images)
    
    print("Done! Created placeholder images for all classes.")
    print(f"Created {len(classes) * num_train_images} training images")
    print(f"Created {len(classes) * num_test_images} test images")

if __name__ == "__main__":
    main() 