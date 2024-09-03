import csv
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_filename_from_path(filepath):
    """
    Extracts the filename from a given file path.

    Args:
        filepath (str): The full path to the file.

    Returns:
        str: The filename without the extension.
    """
    # print("filepathfilepathfilepath", filepath)
    return os.path.splitext(os.path.basename(filepath))[0]


ROOT_DIR = os.path.join("dataset", "v2")

LABEL_DIR = os.path.join(ROOT_DIR, "label")
TRAIN_LABEL_PATH = os.path.join(LABEL_DIR, "train_label.csv")
VALIDATION_LABEL_PATH = os.path.join(LABEL_DIR, "validation_label.csv")
TEST_LABEL_PATH = os.path.join(LABEL_DIR, "test_label.csv")

IMAGES_DIR = os.path.join(ROOT_DIR, "images")
TRAINING_DIR = os.path.join(IMAGES_DIR, "training")
VALIDATION_DIR = os.path.join(IMAGES_DIR, "validation")
TESTING_DIR = os.path.join(IMAGES_DIR, "testing")

ORIGINAL_DIR = os.path.join(ROOT_DIR, "original")
ORIGINAL_TRAINING_DIR = os.path.join(ORIGINAL_DIR, "training")
ORIGINAL_TESTING_DIR = os.path.join(ORIGINAL_DIR, "testing")
ORIGINAL_VALIDATION_DIR = os.path.join(ORIGINAL_DIR, "validation")



def load_images_from_folder(image_folder):
    image_paths = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image_paths.append(image_path)
    return image_paths

# Load images from a folder, filter by .jpg and .png
def load_images(image_folder):
    return [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".jpg") or file.endswith(".png")]

def save_processed_image(image, dest_image_dir, original_filename):
        new_filename = f"{os.path.splitext(original_filename)[0]}{os.path.splitext(original_filename)[1]}"
        save_path = os.path.join(dest_image_dir, new_filename)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise TypeError("Input must be a PIL image or a NumPy array.")
        
        cv2.imwrite(save_path, image)

def extract_filename(file_path):
    """
    Extracts the base filename without extension and underscore suffix.
    Parameters:
    file_path (str): The path of the file.
    Returns:
    str: The base filename without the underscore suffix.
    """
    basename = os.path.basename(file_path)  # Get the base name: "tobi_file.png"
    name, _ = os.path.splitext(basename)    # Split the name and extension: ("tobi_file", ".png")
    name_only = name.split('_')[0]          # Split by underscore and take the first part: "tobi"
    return name_only


def replicate_and_update_row(csv_file_path, match_value, new_value):
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        reader = list(csv.reader(file))
    
    # Find rows where the first column matches match_value
    rows_to_replicate = [row for row in reader if row[0] == match_value]
    
    # Create new rows with updated first column value
    new_rows = []
    for row in rows_to_replicate:
        new_row = row.copy()
        new_row[0] = new_value
        new_rows.append(new_row)
    
    # Append new rows to the original data
    updated_data = reader + new_rows
    
    return updated_data

"""
basename = os.path.basename(file_path)  # Get the base name: "tobi_file.png"
        name, _ = os.path.splitext(basename)    # Split the name and extension: ("tobi_file", ".png")
        name_only = name.split('_')[0]          # Split by underscore and take the first part: "tobi"

"""


def replicate_and_update_row(csv_file_path, names_list):
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        reader = list(csv.reader(file))
    
    new_rows = []
    for name in names_list:
        match_value = name.split('_')[0]
        # Find and replicate rows where the first column matches match_value
        for row in reader:
            if row[0] == match_value:
                new_row = row.copy()
                new_row[0] = name
                new_rows.append(new_row)
    
    # Append new rows to the original data
    updated_data = reader + new_rows
    return updated_data

def write_to_csv(data, destination_file_path):
    # Write the updated data to the destination CSV file
    with open(destination_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def get_trained_model(model_name):
    folder_path = 'trained_model'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, model_name)

# Function to save the model
def save_model(model, model_path):
    n_path = get_trained_model(model_name=model_path)
    torch.save(model.state_dict(), n_path)
    print(f"Model saved to {n_path}")

# Function to load the model
def load_model(model, path):
    n_path = get_trained_model(model_name=path)
    if os.path.exists(n_path):
        model.load_state_dict(torch.load(n_path))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {n_path}")
    else:
        print(f"Model file {n_path} does not exist. Train the model first.")
    return model

# Function to check if the model file exists
def model_file_exist(model_path):
    n_path = get_trained_model(model_name=model_path)
    return os.path.exists(n_path)


def print_metrics(labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(labels, predictions)
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)

# https://www.kaggle.com/code/deathtrooper/pytorch-easy-setup-for-glaucoma-detection-91-8