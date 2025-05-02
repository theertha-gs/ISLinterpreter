"""
Run the entire Ishara sign language pipeline with a single command.

This script will:
1. Check for training data
2. Process images and extract landmarks
3. Train the model
4. Start the application server
"""

import os
import subprocess
import sys
import argparse
import time

def print_header(message):
    """Print a header message"""
    print("\n" + "=" * 80)
    print(message.center(80))
    print("=" * 80 + "\n")

def run_command(command, description, shell=False):
    """Run a command and handle errors"""
    print_header(f"Running: {description}")
    print(f"Command: {command}")
    
    try:
        if shell:
            process = subprocess.run(command, shell=True, check=True)
        else:
            process = subprocess.run(command, check=True)
        
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running {description}: {e}")
        return False

def train_model(args):
    """Run the training pipeline"""
    print_header("Starting Training Pipeline")
    
    # First, check if the script exists
    if os.path.exists("train_isl_model.py"):
        return run_command([sys.executable, "train_isl_model.py"], "Model Training Pipeline")
    
    # If the script doesn't exist, run the individual steps
    print("train_isl_model.py not found, running individual steps...")
    
    if not run_command([sys.executable, "image_to_csv.py"], "Image Preprocessing"):
        if args.synthetic and os.path.exists("create_synthetic_landmarks.py"):
            print("Image processing failed, trying synthetic data generation...")
            if not run_command([sys.executable, "create_synthetic_landmarks.py"], "Synthetic Data Generation"):
                return False
        else:
            return False
    
    if not run_command([sys.executable, "csv_to_model.py"], "Model Training"):
        return False
    
    return True

def run_server(args):
    """Run the application server"""
    print_header("Starting Application Server")
    
    # Check if a port was specified
    port = args.port if args.port else 8000
    
    # Run the server
    cmd = [sys.executable, "app.py"]
    
    print(f"Starting server on port {port}...")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error running server: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run the Ishara Sign Language Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data if image processing fails")
    parser.add_argument("--run", action="store_true", help="Run the application server")
    parser.add_argument("--port", type=int, help="Port to run the server on (default: 8000)")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run everything
    if not (args.train or args.run):
        args.train = True
        args.run = True
    
    # Make sure we're in the correct directory
    if not os.path.exists("app.py"):
        print("❌ Error: This script must be run from the backend directory")
        print(f"Current directory: {os.getcwd()}")
        return False
    
    start_time = time.time()
    
    # Train the model if requested
    if args.train:
        if not train_model(args):
            print("\n❌ Training failed, aborting pipeline")
            return False
    
    # Run the server if requested
    if args.run:
        if not run_server(args):
            print("\n❌ Server failed to start")
            return False
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nTotal pipeline execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 