import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modeltraining.py
from tensorflow.keras.models import load_model

def load_trained_model(model_path):
    """
    Load the trained model from the given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)
import numpy as np

def make_prediction(model, img_array):
    """
    Make predictions using the trained model.
    Returns the predicted class and confidence score.
    """
    prediction = model.predict(img_array)
    class_names = ['NORMAL', 'PNEUMONIA']
    predicted_class = class_names[np.argmax(prediction)]
    confidence_score = np.max(prediction) * 100  # Convert to percentage
    return predicted_class, confidence_score





# Function to create a CNN model using VGG16 as the base
def create_model(input_shape=(128, 128, 3), num_classes=2):
    """
    Creates a CNN model using VGG16 as the base model with transfer learning.
    
    Parameters:
    - input_shape (tuple): Shape of the input images (default is 128x128x3).
    - num_classes (int): Number of output classes (2 in this case: NORMAL, PNEUMONIA).
    
    Returns:
    - model (tensorflow.keras.Model): Compiled CNN model.
    """
    # Load VGG16 without the top layers, pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the custom model on top of the base model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Global average pooling layer
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Function to train the model
def train_model(train_dir, val_dir, batch_size=32, epochs=10):
    """
    Trains the CNN model using the train and validation datasets.
    
    Parameters:
    - train_dir (str): Path to the training dataset directory.
    - val_dir (str): Path to the validation dataset directory.
    - batch_size (int): Number of samples per gradient update (default is 32).
    - epochs (int): Number of epochs to train the model (default is 10).
    
    Returns:
    - history: The training history object.
    """
    # Set up ImageDataGenerators for data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load the data from the directories
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128),
                                                        batch_size=batch_size, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=(128, 128),
                                                    batch_size=batch_size, class_mode='categorical')
    
    # Create the model
    model = create_model(input_shape=(128, 128, 3), num_classes=2)
    
    # Set up callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    
    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator,
                        callbacks=[early_stopping, model_checkpoint])
    
    return model, history

if __name__ == "__main__":
    # Paths to the train and validation directories
    TRAIN_DIR = r"D:\chestxray final\chest_xray\train"  # Replace with the actual path to your training data
    VAL_DIR = r"D:\chestxray final\chest_xray\val"    # Replace with the actual path to your validation data
    
    model, history = train_model(TRAIN_DIR, VAL_DIR, batch_size=32, epochs=10)
    
    print("Training Complete!")
    # Optionally, save the model
    model.save('final_model.h5')
