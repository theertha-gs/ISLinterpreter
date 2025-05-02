"""
This script serves as the main entry point for the training pipeline.
It orchestrates the entire training process from data preparation to model training.
"""

import os
import subprocess
import time
import sys
import json
import shutil
from tqdm import tqdm

def check_training_data():
    """
    Check if the training data is available and properly structured.
    Returns the path to the training data.
    """
    print("Checking for training data...")
    
    # Check in current directory first
    if os.path.exists("training_data") and os.path.isdir("training_data"):
        base_folder = "training_data"
        print(f"Found training data in current directory: {os.path.abspath(base_folder)}")
    # Check in parent directory next
    elif os.path.exists("../training_data") and os.path.isdir("../training_data"):
        base_folder = "../training_data"
        print(f"Found training data in parent directory: {os.path.abspath(base_folder)}")
    else:
        print("❌ Error: No training_data directory found. Please extract your dataset first.")
        return None
    
    # Check if train folder exists
    train_folder = os.path.join(base_folder, "train")
    if not os.path.exists(train_folder):
        print(f"❌ Error: Train folder not found at {train_folder}")
        return None
    
    # Check if train folder contains data
    train_classes = [d for d in os.listdir(train_folder) 
                   if os.path.isdir(os.path.join(train_folder, d))]
    
    if not train_classes:
        print(f"❌ Error: No class folders found in {train_folder}")
        return None
    
    print(f"Found {len(train_classes)} class folders in train directory: {sorted(train_classes)}")
    
    # Check/create test folder
    test_folder = os.path.join(base_folder, "test")
    if not os.path.exists(test_folder):
        print(f"Creating test folder at {test_folder}")
        os.makedirs(test_folder, exist_ok=True)
    
    # Check test folder for classes
    test_classes = [d for d in os.listdir(test_folder) 
                  if os.path.isdir(os.path.join(test_folder, d))]
    
    # Create missing test class folders
    for class_name in train_classes:
        test_class_path = os.path.join(test_folder, class_name)
        if not os.path.exists(test_class_path):
            print(f"Creating test folder for class {class_name}")
            os.makedirs(test_class_path, exist_ok=True)
    
    # Update class_names.json
    all_classes = sorted(train_classes)
    with open("class_names.json", "w") as f:
        json.dump(all_classes, f)
    print(f"Updated class_names.json with {len(all_classes)} classes")
    
    # Ensure csv directory exists
    os.makedirs("csv", exist_ok=True)
    
    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)
    
    return base_folder

def copy_train_samples_to_test(base_folder, test_ratio=0.2):
    """
    Copy a portion of training samples to test directory for validation
    """
    print("Copying samples from train to test directories for validation...")
    
    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")
    
    train_classes = [d for d in os.listdir(train_folder) 
                   if os.path.isdir(os.path.join(train_folder, d))]
    
    total_copied = 0
    
    for class_name in tqdm(train_classes, desc="Copying class samples"):
        train_class_dir = os.path.join(train_folder, class_name)
        test_class_dir = os.path.join(test_folder, class_name)
        
        # Ensure test class directory exists
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get list of image files
        image_files = [f for f in os.listdir(train_class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Skip if not enough files
        if len(image_files) < 5:
            print(f"  Skipping class {class_name}: Not enough samples ({len(image_files)} found)")
            continue
        
        # Calculate number of files to copy
        num_to_copy = max(3, int(len(image_files) * test_ratio))
        
        # If the test directory already has files, check if we need to add more
        existing_test_files = [f for f in os.listdir(test_class_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(existing_test_files) >= num_to_copy:
            print(f"  Class {class_name}: Test directory already has {len(existing_test_files)} samples (sufficient)")
            continue
        
        # Select files to copy (excluding any that are already in test)
        eligible_files = [f for f in image_files if f not in existing_test_files]
        files_to_copy = eligible_files[:num_to_copy - len(existing_test_files)]
        
        # Copy files
        for filename in files_to_copy:
            src = os.path.join(train_class_dir, filename)
            dst = os.path.join(test_class_dir, filename)
            shutil.copy2(src, dst)
        
        total_copied += len(files_to_copy)
        print(f"  Class {class_name}: Copied {len(files_to_copy)} samples to test directory")
    
    print(f"Total samples copied to test directory: {total_copied}")

def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            print(f"\n✅ {script_name} completed successfully")
            return True
        else:
            print(f"\n❌ {script_name} failed with return code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running {script_name}: {e}")
        return False

def main():
    """Main function to orchestrate the training pipeline"""
    start_time = time.time()
    print("\n" + "="*80)
    print("Starting Ishara Sign Language Training Pipeline")
    print("="*80 + "\n")
    
    # Check training data
    base_folder = check_training_data()
    if base_folder is None:
        print("❌ Training aborted: Training data check failed")
        return False
    
    # Check if test directories are empty and copy samples if needed
    test_folder = os.path.join(base_folder, "test")
    test_classes = [d for d in os.listdir(test_folder) 
                  if os.path.isdir(os.path.join(test_folder, d))]
    
    # Check if test classes are empty and copy samples if needed
    empty_test_classes = []
    for class_name in test_classes:
        test_class_dir = os.path.join(test_folder, class_name)
        image_files = [f for f in os.listdir(test_class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) == 0:
            empty_test_classes.append(class_name)
    
    if empty_test_classes:
        print(f"Found {len(empty_test_classes)} empty test class directories")
        copy_train_samples_to_test(base_folder)
    
    # Run the pipeline scripts
    steps = [
        "image_to_csv.py",
        "csv_to_model.py"
    ]
    
    for step in steps:
        if not run_script(step):
            print(f"❌ Training pipeline aborted at step: {step}")
            return False
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nTraining pipeline completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\n✅ Ishara Sign Language model training complete!")
    print("You can now run the application with: python app.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 