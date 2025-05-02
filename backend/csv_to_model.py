"""
This script takes the landmarks CSV files and trains a model to recognize sign language gestures.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import time
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

def load_data():
    """Load and preprocess the training and testing data"""
    try:
        # Load the CSV files
        logger.info("Loading training and testing data...")
        train_csv = os.path.join("csv", "train_landmarks.csv")
        test_csv = os.path.join("csv", "test_landmarks.csv")
        
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Training data not found at {train_csv}. Run image_to_csv.py first.")
        
        if not os.path.exists(test_csv):
            logger.warning(f"Test data not found at {test_csv}. Will use a portion of training data for validation.")
            # Split training data into train and validation
            df_train = pd.read_csv(train_csv)
            train_size = int(0.8 * len(df_train))
            df_test = df_train.iloc[train_size:]
            df_train = df_train.iloc[:train_size]
            logger.info(f"Split training data: {len(df_train)} for training, {len(df_test)} for testing")
        else:
            # Load both training and testing data
            df_train = pd.read_csv(train_csv)
            df_test = pd.read_csv(test_csv)
            logger.info(f"Loaded {len(df_train)} training samples and {len(df_test)} testing samples")
        
        # Convert all labels to strings and ensure they're in the correct format
        df_train['label'] = df_train['label'].astype(str).str.strip()
        df_test['label'] = df_test['label'].astype(str).str.strip()
        
        # Extract features and labels
        feature_columns = [col for col in df_train.columns if col not in ['label', 'image_path']]
        X_train = df_train[feature_columns].values
        X_test = df_test[feature_columns].values
        y_train_labels = df_train['label'].values
        y_test_labels = df_test['label'].values
        
        # Scale the features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save the scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Encode the labels
        logger.info("Encoding labels...")
        label_encoder = LabelEncoder()
        label_encoder.fit(np.concatenate([y_train_labels, y_test_labels]))
        y_train = label_encoder.transform(y_train_labels)
        y_test = label_encoder.transform(y_test_labels)
        
        # Save the label encoder
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
        
        # Get the classes
        classes = label_encoder.classes_
        logger.info(f"Found {len(classes)} classes: {classes}")
        
        return X_train, X_test, y_train, y_test, classes
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_model(input_shape, num_classes):
    """Create and compile the model"""
    model = Sequential([
        # First block
        Dense(1024, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.2),
        
        # Second block
        Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.2),
        
        # Third block
        Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.2),
        
        # Fourth block
        Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training history and save to file"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model/training_history.png")
    print("Saved training history plot to model/training_history.png")

def evaluate_model(model, X_test, y_test, classes):
    """Evaluate the model and save confusion matrix"""
    print("\nEvaluating model on test data...")
    
    # Model evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        y_true_classes, 
        y_pred_classes, 
        target_names=classes,
        digits=4
    )
    print(report)
    
    # Save report to file
    with open("model/classification_report.txt", "w") as f:
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add class labels
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add count labels to the cells
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("model/confusion_matrix.png")
        print("Saved confusion matrix to model/confusion_matrix.png")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")

def main():
    """Main function to train the model"""
    try:
        logger.info("Starting model training...")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, classes = load_data()
        
        # Create and compile model
        input_shape = (X_train.shape[1],)
        num_classes = len(classes)
        model = create_model(input_shape, num_classes)
        
        # Train the model with more epochs and callbacks
        logger.info("Training model...")
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,  # Increased epochs
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        logger.info("Saving model...")
        model.save('models/isl_model.h5')
        
        # Save class names
        with open('class_names.json', 'w') as f:
            json.dump(classes.tolist(), f)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=classes))
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.close()
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()