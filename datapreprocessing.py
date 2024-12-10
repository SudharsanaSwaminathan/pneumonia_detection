import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image
# datapreprocessing.py
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Preprocess the image for prediction.
    - Loads the image.
    - Resizes it to target size.
    - Converts it to an array and preprocesses it (for VGG16).
    """
    # Load the image with target size
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Add a batch dimension (required for models)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image (for VGG16, but can be modified if using a different model)
    img_array = preprocess_input(img_array)
    
    return img_array


# Function to load data from train and test directories
def load_data(data_dir):
    """
    Load images and labels from the dataset directory (both train and test).
    
    Parameters:
    - data_dir (str): Path to the parent directory containing 'train' and 'test' directories.
    
    Returns:
    - images (list): List of image arrays
    - labels (list): List of corresponding labels
    - class_names (list): List of class names (for reference)
    """
    images = []
    labels = []
    class_names = ['NORMAL', 'PNEUMONIA']  # Assuming the dataset contains these two classes
    
    # Loop through train and test directories
    for subset in ['train', 'test']:  # 'train' and 'test' directories
        subset_dir = os.path.join(data_dir, subset)
        for class_name in class_names:  # 'NORMAL' and 'PNEUMONIA'
            class_dir = os.path.join(subset_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Check for valid image file extension
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = Image.open(img_path).convert("L")  # Convert to grayscale
                        img = img.resize((128, 128))  # Resize to 128x128
                        images.append(np.array(img))
                        labels.append(0 if class_name == 'NORMAL' else 1)  # Label 0 for NORMAL, 1 for PNEUMONIA
                    else:
                        print(f"Skipping invalid file: {img_path}")
                except (IOError, FileNotFoundError) as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

# Function to preprocess the data
def preprocess_data(images, labels):
    """
    Normalize the images and one-hot encode the labels.
    
    Parameters:
    - images (numpy array): Array of image data
    - labels (numpy array): Array of corresponding labels
    
    Returns:
    - images (numpy array): Normalized image data
    - labels (numpy array): One-hot encoded labels
    """
    # Normalize image data to the range [0, 1]
    images = images / 255.0
    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=2)  # We have 2 classes: NORMAL and PNEUMONIA
    return images, labels

# Function to split the data into training and testing sets
def split_data(images, labels, test_size=0.2):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    - images (numpy array): Image data
    - labels (numpy array): Labels
    - test_size (float): Proportion of the dataset to include in the test split
    
    Returns:
    - x_train, x_test, y_train, y_test: Training and testing datasets
    """
    return train_test_split(images, labels, test_size=test_size, random_state=42)

if __name__ == "__main__":
    # Example usage
    DATA_DIR = "D:\chestxray final\chest_xray"  # Replace with the actual dataset path containing 'train' and 'test' folders
    images, labels, class_names = load_data(DATA_DIR)
    print(f"Loaded {len(images)} images from {len(class_names)} classes.")
    
    images, labels = preprocess_data(images, labels)
    x_train, x_test, y_train, y_test = split_data(images, labels)
    print(f"Training data: {x_train.shape}, Testing data: {x_test.shape}")
