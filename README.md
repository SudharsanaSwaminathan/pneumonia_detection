# Chest X-ray Diagnosis and Report Generation

This project uses machine learning and natural language processing (NLP) to diagnose pneumonia from chest X-ray images. The Streamlit app allows users to upload a chest X-ray image, get a prediction about the presence of pneumonia, and generate a detailed medical report. The report is then saved as a downloadable PDF.
## Dataset

The dataset used for training the model is the **Chest X-ray Images (Pneumonia)** dataset, which is publicly available on Kaggle. It contains chest X-ray images that are labeled as either **normal** or **pneumonia**. The dataset consists of images of patients with bacterial pneumonia and normal healthy lungs, and it is used for binary classification.

- **Dataset Source**: [Chest X-ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,856 images (normal and pneumonia)
- **Image Size**: 1024 x 1024 pixels (may vary depending on preprocessing)
- **Categories**: 
  - **Normal**: Healthy lungs with no pneumonia.
  - **Pneumonia**: Includes both bacterial and viral pneumonia.

### Dataset Structure

The dataset is organized into two main folders:
- **train**: Training data, including subfolders for 'NORMAL' and 'PNEUMONIA'.
- **test**: Testing data, also separated into 'NORMAL' and 'PNEUMONIA'.

Each image is in JPG format and represents an X-ray scan of a patient's chest.

### How the Dataset is Used

In this project, the images are used for training and testing a deep learning model to predict whether a chest X-ray image shows signs of pneumonia. The model is trained using the **train** set, and its performance is evaluated using the **test** set.

The images are preprocessed before being fed into the model to ensure they meet the input size requirements (e.g., resizing to 128x128 pixels). The model then classifies each image into one of the two categories: **Normal** or **Pneumonia**.



## Features
- Upload a chest X-ray image in `.jpg`, `.jpeg`, or `.png` format.
- Preprocess the image for prediction.
- Make predictions using a pre-trained deep learning model.
- Generate a medical report using GPT-Neo and the prediction results.
- Download the report as a PDF.

## File Structure
- **`app.py`**: The main Streamlit application that integrates all functionalities.
- **`datapreprocessing.py`**: Contains the image preprocessing steps, including resizing and normalizing the image.
- **`modeltraining.py`**: Loads the trained model and makes predictions based on the uploaded image.
- **`reportgenerationpdf.py`**: Generates a detailed medical report and saves it as a PDF file.
  
## Technologies Used
- **Streamlit**: For building the interactive web app.
- **TensorFlow / Keras**: For loading the pre-trained model and making predictions.
- **GPT-Neo**: A transformer-based language model used to generate detailed medical reports.
- **ReportLab**: For generating and saving the medical report as a PDF.

## Prerequisites
Before running the project, ensure you have the following libraries installed:

```bash
pip install streamlit tensorflow transformers reportlab Pillow numpy
