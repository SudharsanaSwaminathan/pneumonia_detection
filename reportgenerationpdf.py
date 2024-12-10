import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# In reportgenerationpdf.py
# In reportgenerationpdf.py
def generate_report(predicted_class, confidence_score):
    print(f"generate_report called with: {predicted_class}, {confidence_score}")
    # (Rest of the function)

def generate_report(predicted_class, confidence_score):
    """
    Generate a detailed medical report based on the prediction and confidence score.
    """
    report = (
        f"Chest X-ray Report\n\n"
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence_score:.2f}%\n\n"
        f"Details:\n"
        f"This report is generated based on a chest X-ray image."
        f" The prediction indicates {predicted_class}, with a confidence of {confidence_score:.2f}%."
    )
    return report




# Load the trained model
model = load_model('best_model.h5')

# Load GPT-Neo model and tokenizer for report generation
gpt_model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPTNeoForCausalLM.from_pretrained(gpt_model_name)

# Function to preprocess the image for prediction
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for VGG16
    return img_array

# Function to generate a report using GPT-Neo
def generate_gpt_report(predicted_class, confidence_score):
    """
    Generate a detailed report using GPT-Neo based on the prediction.
    """
    prompt = f"Generate a detailed medical report for a chest X-ray. The prediction is {predicted_class} with a confidence of {confidence_score:.2f}%. Provide a brief explanation of the condition."
    
    # Encode the prompt and generate text using GPT-Neo
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = gpt_model.generate(inputs['input_ids'], max_length=250, num_return_sequences=1)
    
    # Decode the generated text
    generated_report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_report

# Function to generate a prediction report for the X-ray image

    

# Function to save the report as a PDF
def save_report_as_pdf(predicted_class, confidence_score, gpt_report, pdf_filename="Xray_Report.pdf"):
    """
    Save the generated report as a PDF.
    """
    # Create a PDF file with the generated report
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter  # Letter size is 8.5 x 11 inches
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, f"Chest X-ray Report: {predicted_class} ({confidence_score:.2f}%)")
    
    # Prediction details
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Prediction: {predicted_class}")
    c.drawString(100, height - 100, f"Confidence: {confidence_score:.2f}%")
    
    # Add the generated report text
    text_object = c.beginText(100, height - 140)
    text_object.setFont("Helvetica", 10)
    text_object.setTextOrigin(100, height - 140)
    text_object.textLines(gpt_report)
    
    c.drawText(text_object)
    
    # Save the PDF
    c.save()


