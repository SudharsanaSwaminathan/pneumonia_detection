import streamlit as st
import os
from PIL import Image

# Import functions from your existing files
from datapreprocessing import preprocess_image  # Function to preprocess the image
from modeltraining import load_trained_model, make_prediction  # Model loading and prediction
from reportgenerationpdf import generate_report, save_report_as_pdf  # Report generation and saving as PDF

# Streamlit UI components
st.title("Chest X-ray Diagnosis and Report Generation")
st.write("Upload a chest X-ray image, and we'll generate a diagnosis report and save it as a PDF.")

# File uploader
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Chest X-ray Image", use_column_width=True)
    
    # Save uploaded image temporarily
    img_path = os.path.join("temp_dir", uploaded_file.name)
    os.makedirs("temp_dir", exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Prediction and report generation on button click
    if st.button("Generate Report"):
        # Step 1: Preprocess the image
        img_array = preprocess_image(img_path)  # Assuming this function is in dataprocessing.py
        
        # Step 2: Load the trained model and make a prediction
        model = load_trained_model('best_model.h5')  # Assuming this function is in model_training.py
        predicted_class, confidence_score = make_prediction(model, img_array)  # Adjust if needed
        
        # Step 3: Generate a detailed report using your existing function
        gpt_report = generate_report(predicted_class, confidence_score)  # Assuming this function is in report_generation.py
        
        # Display the generated report
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence_score:.2f}%")
        st.write("Generated Report:")
        st.text_area("Medical Report", gpt_report, height=300)
        
        # Step 4: Save the report as a PDF
        pdf_filename = f"{uploaded_file.name}_report.pdf"
        save_report_as_pdf(predicted_class, confidence_score, gpt_report, pdf_filename)  # Assuming this function is in report_generation.py
        
        # Provide a download link for the PDF
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=pdf_filename,
                mime="application/pdf"
            )
