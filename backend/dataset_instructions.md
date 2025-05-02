# Dataset Organization Instructions

This guide explains how to organize your sign language dataset for use with the Ishara Sign Language Translator.

## Option 1: Using the Dataset Extraction Tool

If you have a zip file containing sign language images, you can use our automatic extraction script:

```
python extract_dataset.py path/to/your/dataset.zip
```

### How it works:

1. The script extracts the zip file to a temporary directory
2. It intelligently looks for classes based on directory structure or filenames
3. It organizes images into train/test folders with an 80/20 split
4. It creates/updates class_names.json with the detected sign classes
5. It cleans up temporary files when done (unless you use the `--keep-temp` option)

### Command-line options:

- `--train-ratio`: Adjust the train/test split ratio (default: 0.8)  
  Example: `python extract_dataset.py dataset.zip --train-ratio 0.75`

- `--keep-temp`: Keep the temporary extraction directory after processing  
  Example: `python extract_dataset.py dataset.zip --keep-temp`  
  This is useful if the automatic organization fails and you need to manually organize the files.

### Troubleshooting dataset extraction:

If the automatic extraction cannot organize your dataset properly, the script will:
1. Keep the temporary directory with the extracted files
2. Show the path where files are extracted
3. Provide instructions for manual organization

You can examine the extracted files and organize them manually according to the instructions in Option 2 below.

## Option 2: Manual Organization

If you prefer to organize your dataset manually, follow these steps:

### Step 1: Create the Directory Structure

Create the following directory structure in the backend folder:

```
backend/training_data/
  train/
  test/
```

### Step 2: Organize Your Dataset

For each sign language gesture or letter (A, B, C, etc.) in your dataset:

1. Create a folder for the sign in both train and test directories:
   ```
   backend/training_data/train/A/
   backend/training_data/test/A/
   ```

2. Distribute your images:
   - Place approximately 80% of the images for each sign in the corresponding `train` folder
   - Place the remaining 20% in the corresponding `test` folder

For example, if you have 100 images for sign "A", place 80 in `training_data/train/A/` and 20 in `training_data/test/A/`.

### Step 3: Update class_names.json

Create or update the `class_names.json` file in the backend directory with a list of all the sign classes in your dataset:

```json
["A", "B", "C", ...]
```

Include all sign classes in alphabetical or logical order.

## Training the Model

Once your dataset is organized, you can train the model:

```
python train_isl_model.py
```

This will:
1. Process all images to extract hand landmarks using MediaPipe
2. Train a Random Forest model on the extracted features
3. Save the model and related files

## Running the Application

After training is complete, run the application:

```
python app.py
```

## Tips for Better Results

- **Image Quality**: Make sure your images clearly show hand signs against a contrasting background
- **Image Quantity**: Aim for at least 100-200 images per sign class for good accuracy
- **Image Variety**: Include different lighting conditions, backgrounds, and slight variations in hand position
- **Balanced Classes**: Try to have roughly the same number of images for each sign class

## Troubleshooting

- If you encounter issues with MediaPipe not detecting hands in some images, try improving image contrast or lighting
- If accuracy is low, consider adding more training images or adjusting the model parameters in `csv_to_model.py` 