"""
This script extracts a zip file containing a sign language dataset
and organizes it into the training_data directory structure.
"""
import os
import zipfile
import shutil
import json
import argparse
import random
import glob
from tqdm import tqdm

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction completed to {extract_to}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def print_directory_structure(directory, indent=0):
    """Print the structure of a directory for debugging purposes"""
    print(' ' * indent + os.path.basename(directory) + '/')
    
    # List all files and directories
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            print_directory_structure(path, indent + 2)
        else:
            print(' ' * (indent + 2) + item)

def detect_image_structure(source_dir):
    """Detect the structure of the image dataset"""
    # First, look for standard patterns of dataset organization
    
    print(f"Analyzing directory structure in: {source_dir}")
    print("Top-level contents:")
    for item in os.listdir(source_dir):
        path = os.path.join(source_dir, item)
        if os.path.isdir(path):
            print(f"  Directory: {item} (contains {len(os.listdir(path))} items)")
        else:
            print(f"  File: {item}")
    
    # Count images in source directory and subdirectories
    all_images = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                all_images.append(os.path.join(root, file))
    
    print(f"\nFound {len(all_images)} total images in the dataset")
    
    # Check for common dataset patterns
    possible_class_dirs = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            subdir_image_count = len(glob.glob(os.path.join(item_path, "*.png")) + 
                                glob.glob(os.path.join(item_path, "*.jpg")) + 
                                glob.glob(os.path.join(item_path, "*.jpeg")))
            if subdir_image_count > 0:
                # This could be a class directory if it contains images directly
                possible_class_dirs.append((item, item_path, subdir_image_count))
    
    if possible_class_dirs:
        print("\nPossible sign class directories:")
        for class_name, path, count in possible_class_dirs:
            print(f"  {class_name}: {count} images")
    
    return possible_class_dirs, all_images

def organize_dataset(source_dir, train_ratio=0.8, keep_temp=False):
    """
    Organize the dataset into train/test folders
    
    This function tries to intelligently organize images by looking for 
    directories that might represent sign classes
    """
    print("\nOrganizing dataset...")
    
    # Create necessary directories
    os.makedirs("training_data/train", exist_ok=True)
    os.makedirs("training_data/test", exist_ok=True)
    
    # Detect dataset structure
    class_dirs, all_images = detect_image_structure(source_dir)
    
    # If we couldn't find clear class directories, try alternative approaches
    if not class_dirs:
        print("\nNo obvious class directories found. Let's try different approaches...")
        
        # APPROACH 1: Look for parent directories that might have class subfolders
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                # Check for subdirectories that could be classes
                inner_class_dirs = []
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        # Check if it might be a class directory (short name)
                        inner_image_count = len(glob.glob(os.path.join(subitem_path, "*.png")) + 
                                               glob.glob(os.path.join(subitem_path, "*.jpg")))
                        if inner_image_count > 0:
                            inner_class_dirs.append((subitem, subitem_path))
                
                if inner_class_dirs:
                    print(f"Found {len(inner_class_dirs)} potential class directories inside {item}")
                    class_dirs = inner_class_dirs
                    break
        
        # APPROACH 2: If still no class directories, try inferring from filenames
        if not class_dirs and all_images:
            print("Trying to infer classes from image filenames...")
            
            # First, print some example filenames
            sample_size = min(5, len(all_images))
            print(f"Sample filenames (from {len(all_images)} images):")
            for i in range(sample_size):
                print(f"  {os.path.basename(all_images[i])}")
            
            # Try to infer classes from filenames
            classes = set()
            for img_path in all_images:
                # Get filename without extension
                filename = os.path.splitext(os.path.basename(img_path))[0]
                
                # Different approaches to extract class information:
                
                # 1. First character of filename (typically for single letter signs like A, B, C)
                if len(filename) > 0 and filename[0].isalpha():
                    classes.add(filename[0].upper())
                
                # 2. Characters before first underscore/delimiter
                parts = filename.split('_')
                if len(parts) > 1 and parts[0].isalnum():
                    classes.add(parts[0].upper())
                
                # 3. Characters before first digit
                for i, char in enumerate(filename):
                    if char.isdigit():
                        potential_class = filename[:i].strip().upper()
                        if potential_class and potential_class.isalpha():
                            classes.add(potential_class)
                        break
            
            if classes:
                print(f"Inferred {len(classes)} potential classes from filenames: {sorted(list(classes))}")
                
                # Distribute images to classes
                for cls in classes:
                    os.makedirs(f"training_data/train/{cls}", exist_ok=True)
                    os.makedirs(f"training_data/test/{cls}", exist_ok=True)
                
                # Copy images to appropriate folders
                successful_copies = 0
                for img_path in tqdm(all_images, desc="Organizing images"):
                    filename = os.path.splitext(os.path.basename(img_path))[0]
                    assigned_class = None
                    
                    # Try different ways to match file to class
                    if len(filename) > 0 and filename[0].upper() in classes:
                        assigned_class = filename[0].upper()
                    
                    # Try splitting by underscore
                    if not assigned_class:
                        parts = filename.split('_')
                        if len(parts) > 1 and parts[0].upper() in classes:
                            assigned_class = parts[0].upper()
                    
                    # Try part before first digit
                    if not assigned_class:
                        for i, char in enumerate(filename):
                            if char.isdigit():
                                potential_class = filename[:i].strip().upper()
                                if potential_class in classes:
                                    assigned_class = potential_class
                                break
                    
                    if assigned_class:
                        # Determine train or test
                        is_train = random.random() < train_ratio
                        dest_dir = f"training_data/{'train' if is_train else 'test'}/{assigned_class}"
                        
                        # Copy the image
                        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                        shutil.copy2(img_path, dest_path)
                        successful_copies += 1
                
                # Update class_names.json
                with open("class_names.json", "w") as f:
                    json.dump(sorted(list(classes)), f)
                
                print(f"Successfully copied {successful_copies} out of {len(all_images)} images")    
                print(f"Organized images into {len(classes)} classes: {sorted(list(classes))}")
                
                # Check if any images were successfully copied
                if successful_copies > 0:
                    return True
                else:
                    print("Could not match any images to the inferred classes")
                    return False
            else:
                print("Could not infer any classes from filenames")
        
        # APPROACH 3: Manual assignment based on image content
        if not class_dirs:
            print("\nCould not automatically organize the dataset.")
            print("The extracted files remain in the following directory:")
            print(f"  {os.path.abspath(source_dir)}")
            print("\nPlease manually organize the images into the following structure:")
            print("  training_data/train/A/  (for letter A images)")
            print("  training_data/train/B/  (for letter B images)")
            print("  ... and so on for other classes")
            print("  training_data/test/A/  (for letter A test images)")
            print("  training_data/test/B/  (for letter B test images)")
            print("  ... and so on for other classes")
            
            # We're keeping the extract directory for manual processing
            if not keep_temp:
                print("\nThe temporary directory will NOT be deleted so you can manually organize the files.")
                print(f"Location: {os.path.abspath(source_dir)}")
            
            return False
    
    # Process each class directory
    print(f"\nFound {len(class_dirs)} potential class directories")
    classes = []
    
    for class_info in class_dirs:
        if len(class_info) == 2:
            class_name, class_dir = class_info
        else:
            class_name, class_dir, _ = class_info
            
        classes.append(class_name)
        os.makedirs(f"training_data/train/{class_name}", exist_ok=True)
        os.makedirs(f"training_data/test/{class_name}", exist_ok=True)
        
        # Find all images in this class directory
        images = []
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    images.append(os.path.join(root, file))
        
        print(f"Found {len(images)} images for class {class_name}")
        
        if not images:
            continue
            
        # Shuffle images for random split
        random.shuffle(images)
        
        # Determine split point
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Copy images to train folder
        for img in tqdm(train_images, desc=f"Processing {class_name} (train)"):
            dest_path = os.path.join(f"training_data/train/{class_name}", os.path.basename(img))
            shutil.copy2(img, dest_path)
        
        # Copy images to test folder
        for img in tqdm(test_images, desc=f"Processing {class_name} (test)"):
            dest_path = os.path.join(f"training_data/test/{class_name}", os.path.basename(img))
            shutil.copy2(img, dest_path)
    
    # Update class_names.json
    if classes:
        with open("class_names.json", "w") as f:
            json.dump(sorted(classes), f)
        
        print(f"Dataset organized into {len(classes)} classes: {sorted(classes)}")
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract and organize a sign language dataset from a zip file")
    parser.add_argument("zip_file", help="Path to the zip file containing the dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                        help="Ratio of train/test split (default: 0.8)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep the temporary extraction directory after processing")
    
    args = parser.parse_args()
    
    # Create a temporary directory for extraction
    extract_dir = "temp_extract"
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract the zip file
    if not extract_zip(args.zip_file, extract_dir):
        return False
    
    # Organize the dataset
    success = organize_dataset(extract_dir, args.train_ratio, args.keep_temp)
    
    # Clean up temporary directory if not needed
    if success and not args.keep_temp:
        try:
            shutil.rmtree(extract_dir)
            print(f"Cleaned up temporary directory {extract_dir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {e}")
    elif not success or args.keep_temp:
        print(f"\nKept temporary directory: {os.path.abspath(extract_dir)}")
        print("You can examine the files and organize them manually if needed.")
    
    if success:
        print("\nNext steps:")
        print("1. Run image_to_csv.py to process the images and extract landmarks")
        print("2. Run csv_to_model.py to train a model")
        print("   - Or, simply run train_isl_model.py to do both steps")
    else:
        print("\nOnce you have manually organized your dataset, run:")
        print("python train_isl_model.py")
    
    return success

if __name__ == "__main__":
    main() 