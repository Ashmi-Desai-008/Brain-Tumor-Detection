import os
from PIL import Image
import streamlit as st
from keras.models import load_model
import numpy as np

# Get the current directory
current_directory = os.getcwd()

# Define the path to the model file relative to the current directory
MODEL_PATH = os.path.join(current_directory, "brain_tumor_model.keras")

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((128, 128))
    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("Brain Tumor Detection App")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the uploaded image
        processed_image = preprocess_image(image)

        # Load the pre-trained CNN model
        model = load_model(MODEL_PATH)

        # Make predictions using the pre-trained model
        prediction = model.predict(processed_image)
        # Assuming your model predicts probabilities for two classes
        # You can customize this according to your model's output
        if prediction[0][0] > 0.5:
            st.write("Prediction: Tumor Detected")
        else:
            st.write("Prediction: No Tumor Detected")

if __name__ == "__main__":
    main()
