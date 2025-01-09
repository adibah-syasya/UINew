
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import cv2

def apply_clahe(uploaded_file):
    """
    This function takes an image and performs clahe by cv2 on the image.
    Parameter:
        image


    Returns:
        A contrast enhanced image (PIL.Image.Image)
    """    
    try:
        # Read the uploaded file into a NumPy array
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        # Decode the image from the bytes
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the image. Please upload a valid image file.")

        # Convert to LAB color space
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split into L, A, and B channels
        l, a, b = cv2.split(img_lab)
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=3.0)
        # Apply CLAHE to the L channel
        clahe_img = clahe.apply(l)
        # Merge the CLAHE enhanced L channel back with A and B channels
        merge_lab = cv2.merge((clahe_img, a, b))
        # Convert back to RGB color space
        image_lab_updated = cv2.cvtColor(merge_lab, cv2.COLOR_LAB2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_lab_updated)
        return pil_image
    except Exception as e:
        raise RuntimeError(f"Error processing the image: {str(e)}")




def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 255.0) 

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)  # Predict probabilities for 8 classes
    index = np.argmax(prediction)     # Get the index of the highest probability
    class_name = class_names[index]   # Retrieve the class name based on the index
    confidence_score = prediction[0][index]  # Retrieve the confidence score for the predicted class

    
    #prediction = model.predict(data)
    #index = np.argmax(prediction)
    #index = 0 if prediction[0][0] > 0.95 else 1
   # class_name = class_names[index]
    #confidence_score = prediction[0][index]

    return class_name , confidence_score
