import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from PIL import ImageFile

# To avoid the UnidentifiedImageError, we can use this workaround
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to plot sample images from the dataset
def plot_sample_images(data_dir, class_names=['NORMAL', 'PNEUMONIA'], num_samples=5):
    """
    Display a few sample images from each class.
    
    Parameters:
    - data_dir (str): Path to the dataset directory
    - class_names (list): List of class names ('NORMAL' and 'PNEUMONIA')
    - num_samples (int): Number of sample images to display per class
    """
    plt.figure(figsize=(10, 5))
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, 'train', class_name)
        sample_images = os.listdir(class_dir)[:num_samples]
        
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = image.load_img(img_path, target_size=(128, 128))
                plt.subplot(len(class_names), num_samples, i * num_samples + j + 1)
                plt.imshow(img)
                plt.title(class_name)
                plt.axis('off')
            except (IOError, FileNotFoundError, OSError) as e:
                print(f"Skipping invalid or corrupt image {img_path}: {e}")
    
    plt.tight_layout()
    plt.show()

# Function to plot class distribution
def plot_class_distribution(data_dir):
    """
    Plot the distribution of images across different classes in the training set.
    
    Parameters:
    - data_dir (str): Path to the dataset directory
    """
    class_counts = {'NORMAL': 0, 'PNEUMONIA': 0}
    
    for class_name in class_counts.keys():
        class_dir = os.path.join(data_dir, 'train', class_name)
        class_counts[class_name] = len(os.listdir(class_dir))
    
    # Create a bar plot for the class distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.show()

if __name__ == "__main__":
    # Example usage
    DATA_DIR = "D:\chestxray final\chest_xray"  # Replace with the actual dataset path
    plot_sample_images(DATA_DIR)
    plot_class_distribution(DATA_DIR)
