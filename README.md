# Chest X-ray Diagnosis and Report Generation

This project uses machine learning and natural language processing (NLP) to diagnose pneumonia from chest X-ray images. The Streamlit app allows users to upload a chest X-ray image, get a prediction about the presence of pneumonia, and generate a detailed medical report. The report is then saved as a downloadable PDF.

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
