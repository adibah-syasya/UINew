import streamlit as st
from keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
from util3 import classify, apply_clahe
import time, os

# Set title
st.title('Gastrointestinal Disease Classification')
st.subheader("University Malaya Medical Center")
st.write("This application is designed specifically to predict four gastrointestinal (GI) tract diseases :- Angiodysplasia, Lymphangiectasia, Polyps, Ulcerative-colitis")
# Set header
st.subheader('Please upload an endoscopy image')

# Upload file
file = st.file_uploader('')
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}
if file is not None:
    # Validate file extension
    file_extension = os.path.splitext(file.name)[1].lower()[1:]  # Get the file extension without the dot
    if file_extension not in ALLOWED_EXTENSIONS:
        print("Error loading the image. Please make sure image is in jpeg, jpg, or png format")
        st.error("Unsupported file format. Please upload a JPEG, JPG, or PNG image.")
        st.stop()

# Load classifier with error handling
try:
    #model = load_model("C:/Users/Public/AcademicProject/Training_Model/densenet_2.h5")
    model = load_model("densenet_ft_5.h5")
    #model = load_model("C:/Users/Public/AcademicProject/1)New/fine-tuning/densenet_ft_5.h5")
except Exception as e:
    print("Error loading the model")
    st.error(f"Error loading the model: {e}")
    st.stop()
# Load class names with error handling
try:
    #with open("C:/Users/Public/AcademicProject/Interface/UINew/labels2.txt", 'r') as f:
    with open("./labels2.txt", 'r') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
except FileNotFoundError:
    print("Labels file not found. Please make sure 'labels2.txt' is in the correct location.")
    st.error("Labels file not found. Please make sure 'labels2.txt' is in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"Error reading labels file: {e}")
    st.stop()

# Display image and classify if a file is uploaded
if file is not None:
    try:
        start_time = time.time()
        # Perform CLAHE
        enhanced_image = apply_clahe(file)

        # Load and display the original image
        image = Image.open(file).convert('RGB')
        clahe_time = time.time() - start_time
        print(f"Time for CLAHE processing: {clahe_time:.2f} seconds")

        # Classify the image
        try:
            start_time = time.time()
            class_name, conf_score = classify(enhanced_image, model, class_names)
            st.write("## {}".format(class_name))
            st.write("### Score: {}%".format(int(conf_score * 1000) / 10))
            prediction_time = time.time() - start_time
            print(f"Time for prediction: {prediction_time:.2f} seconds")
            st.image(image, caption="Original Image", use_column_width=True)
            st.image(enhanced_image, caption="Enhanced Image (CLAHE)", use_column_width=True)

        except Exception as e:
            st.error(f"Error during classification: {e}")

    except UnidentifiedImageError:
        st.error("Uploaded file is not a valid image. Please upload a JPEG or PNG image.")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

else:
    st.info("Please upload an image to enhance.")

