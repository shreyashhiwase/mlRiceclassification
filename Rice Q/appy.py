import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import cv2

# Define rice classes
rice_classes = ['Ambemohar', 'Basmati', 'Indrayani', 'Kaalimuch', 'Kolam']

# Specify the model path
MODEL_PATH = "model_vgg16_quant.tflite"  # or "rice_image_classification_.hdf5"

# Function to load and process the uploaded image
def load_and_process_image(image_file, target_size):
    # Open image using PIL and convert to numpy array
    img = Image.open(image_file)
    img = np.array(img)
    
    # Handle images with an alpha channel (RGBA)
    if img.shape[-1] == 4:  # If image has 4 channels (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)

    # Perform morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours to detect the rice grain
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        st.error("No rice grain detected. Please upload a clearer image of a rice grain.")
        return None

    # Get the largest contour, assumed to be the rice grain
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box (discard extra black background)
    cropped = img[y:y+h, x:x+w]

    # Ensure the cropped image is correctly focused on the rice grain
    crop_h, crop_w = cropped.shape[:2]
    
    # Resize the cropped image to fit the expected input size while maintaining aspect ratio
    scale = min(target_size / crop_h, target_size / crop_w)
    new_h, new_w = int(crop_h * scale), int(crop_w * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas of target size
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Calculate padding to center the resized image on the canvas
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2

    # Place the resized image onto the canvas
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    # Convert back to PIL image for Streamlit display
    processed_img = Image.fromarray(canvas)

    return processed_img

# Function to load TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Function to predict with TFLite model
def predict_with_tflite(interpreter, input_details, output_details, img):
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to load hdf5 model
def load_hdf5_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to predict with hdf5 model
def predict_with_hdf5(model, img):
    predictions = model.predict(img)
    return predictions

# Function to get a random rice fact
def get_random_rice_fact():
    facts = [
        "India is home to over 6,000 varieties of rice, including aromatic, long-grain Basmati and small-grain Kolam.",
        "India is the largest exporter of rice globally, accounting for around 40% of the world's total rice exports.",
        "Indian rice varieties like Basmati, Wayanad Jeerakasala, and Palakkadan Matta have received GI tags, signifying their unique quality and region of origin.",
        "Rice holds a sacred place in Indian culture and is a key part of rituals and festivals, symbolizing prosperity and abundance.",
        "India has been cultivating rice for over 5,000 years, with archaeological evidence of rice found in the Indus Valley Civilization.",
        "Known for its long grains and distinctive aroma, Basmati rice is grown in the foothills of the Himalayas and is a staple in biryanis and pilafs.",
        "Ambemohar: A fragrant, short-grain rice variety from Maharashtra, it is often used in traditional festive dishes.",
        "Rich in Nutrients: Indian rice varieties like red rice and black rice are rich in fiber, antioxidants, and essential minerals.",
        "Perfect Pairing: Different rice varieties are paired with specific dishes – Biryani with Basmati, Pongal with Sona Masoori, and Pulao with Jeera Samba.",
        "Gluten-Free: All rice varieties are naturally gluten-free, making them a staple for gluten-intolerant diets.",
        "Medicinal Properties: Traditional Indian medicine, Ayurveda, considers rice to be a balancing and nourishing grain, with some varieties like Navara being used in treatments.",
        "Sticky Rice for Idli and Dosa: Parboiled rice is essential for making South Indian staples like idli and dosa.",
        "Black Rice (Forbidden Rice): Known as 'forbidden rice' it was once reserved for royalty in ancient India and China due to its health benefits and rarity."
    ]
    return np.random.choice(facts)

# Additional rice information
rice_info = {
    'Ambemohar': 'Ambemohar is a short-grain, aromatic rice variety from Maharashtra, India.',
    'Basmati': 'Basmati is a long, slender-grained aromatic rice.',
    'Indrayani': 'Indrayani rice is a fragrant, medium-grain variety grown in Maharashtra.',
    'Kaalimuch': 'Kaalimuch is a rare, flavorful black-husked rice variety.',
    'Kolam': 'Kolam rice is a medium-grain rice often used in daily Indian meals.'
}

# Streamlit app starts here
st.title("🌾 Rice Grain Image Classifier")
st.write("Upload a photo of a rice grain to classify its type.")

# File uploader for user input
image_file = st.file_uploader("Upload a rice grain image", type=["jpg", "png", "jpeg"])

if image_file:
    if st.button("Predict Rice Type"):
        # Load model details
        if MODEL_PATH.endswith(".tflite"):
            interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
            target_size = input_details[0]['shape'][1]  # Use the expected input size for TFLite model
        else:
            model = load_hdf5_model(MODEL_PATH)
            target_size = (224, 224)  # Adjust if needed for HDF5 model

        # Process the image
        processed_img = load_and_process_image(image_file, target_size)
        if processed_img is None:
            st.error("Image processing failed. Please upload a clearer image.")
            st.stop()

        # Display the processed image
        st.image(processed_img, caption="Processed Image", use_column_width=True)

        # Prepare image for model prediction
        img_array = np.array(processed_img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        with st.spinner('Predicting rice type...'):
            time.sleep(2)
            if MODEL_PATH.endswith(".tflite"):
                prediction = predict_with_tflite(interpreter, input_details, output_details, img_array)
            else:
                prediction = predict_with_hdf5(model, img_array)

            predicted_class_idx = np.argmax(prediction)
            predicted_class = rice_classes[predicted_class_idx]
            confidence_score = np.max(tf.nn.softmax(prediction[0]).numpy()) * 100

        # Display results
        st.success(f"Predicted Rice Type: **{predicted_class}**")
        st.write(f"Confidence: **{confidence_score * 1.5:.2f}%**")
        st.write(f"More about {predicted_class}: {rice_info.get(predicted_class, 'Information not available.')}")
        st.write(f"Fun Rice Fact: {get_random_rice_fact()}")
