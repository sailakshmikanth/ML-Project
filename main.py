import os
import keras
import numpy as np
from keras.models import load_model
import tensorflow as tf
import streamlit as st


# Set page config for the app
st.set_page_config(page_title='Rash Identifier', layout='wide')

st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f5;  /* Light gray background for a clean look */
        font-family: 'Arial', sans-serif;
        color: #333;
        line-height: 1.6;  /* Improved readability */
    }
    .header {
        text-align: center;
        padding: 30px 0;  /* More padding for a spacious look */
        background-color: #4CAF50;  /* Dark green background */
        color: white;
        border-radius: 5px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);  /* Soft shadow for depth */
        margin-bottom: 30px;  /* Space below the header */
    }
    .upload-button {
        margin: 20px auto;
        padding: 10px 20px;
        background-color: #007BFF;  /* Blue background */
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;  /* Smooth transition */
    }
    .upload-button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
    }
    .file-uploader {
        margin: 20px auto;  /* Center the uploader */
        border: 2px dashed #007BFF;  /* Dashed border */
        border-radius: 5px;
        padding: 20px;
        text-align: center;  /* Center text */
        background-color: #fff;  /* White background */
        box-shadow: 0 1px 5px rgba(0,0,0,0.1);  /* Light shadow for elevation */
    }
    .result {
        padding: 20px;
        margin-top: 20px;
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);  /* Soft shadow */
        border-left: 5px solid #4CAF50;  /* Green left border for visual emphasis */
    }
    .result h4 {
        margin-top: 0;  /* Remove top margin for heading */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Using Markdown for the header
st.markdown('<h1 class="header">Rash Identifier</h1>', unsafe_allow_html=True)

rash_names = ['Dermatitis', 'Eczema', 'RingWorm']

# Links for more information about each disease
disease_links = {
    'Dermatitis': 'https://www.mayoclinic.org/diseases-conditions/dermatitis/symptoms-causes/syc-20351426',
    'Eczema': 'https://www.aad.org/public/diseases/eczema',
    'RingWorm': 'https://www.healthline.com/health/ringworm'  # Updated link
}

# Load the trained model
rashModel = load_model('Rash Identification.h5')

def classifyImages(image_path):
    # Load and preprocess the image
    inputImageTesting = tf.keras.utils.load_img(image_path, target_size=(200, 200))
    inputImageArray = tf.keras.utils.img_to_array(inputImageTesting)
    inputImageExpDimen = tf.expand_dims(inputImageArray, 0)  # Expand dimensions for model input

    # Make prediction
    prediction = rashModel.predict(inputImageExpDimen)
    result = tf.nn.softmax(prediction[0])

    # Get the class with the highest probability
    if np.max(result) < 0.5:
        predicted_class = "non-rash"
    else:
        predicted_class = rash_names[np.argmax(result)]

   # predicted_class = rash_names[np.argmax(result)]
    print(predicted_class)
    # Classification output with links for more information
    if predicted_class == "Dermatitis":
        output = (f"<h4>Dermatitis</h4>"
                  "A general term for skin inflammation, often causing redness, swelling, and itching. "
                  "It can be triggered by allergies, irritants, or genetic factors. "
                  "Immediate attention is needed if there is severe pain, signs of infection (like pus, swelling, or fever), "
                  "or if symptoms rapidly worsen.<br><br>"
                  "<strong>Causes:</strong> Common causes include contact with allergens like soaps, detergents, or fragrances, "
                  "prolonged exposure to water, certain foods, and stress.<br><br>"
                  "<strong>General Advice:</strong> Keep the skin moisturized, avoid known triggers, and consider using hypoallergenic products. "
                  "Consult a dermatologist if symptoms persist or worsen.<br><br>"
                  "<strong>When to Seek Medical Help:</strong> If you experience spreading of redness, severe pain, or signs of infection like fever, "
                  "or if you have difficulty managing the itching.<br><br>"
                  f"<a href='{disease_links['Dermatitis']}'>Learn more</a>")

    elif predicted_class == "Eczema":
        output = (f"<h4>Eczema</h4>"
                  "A chronic skin condition characterized by itchy, dry, and inflamed skin. "
                  "It often flares up due to allergens, stress, or environmental factors. "
                  "Immediate attention is needed if there is severe pain, signs of infection (like pus, swelling, or fever), "
                  "or if symptoms rapidly worsen.<br><br>"
                  "<strong>Causes:</strong> Eczema may be linked to genetic factors, immune system response, irritants like "
                  "household cleaners, certain fabrics, and environmental factors such as temperature changes.<br><br>"
                  "<strong>General Advice:</strong> Keeping skin moisturized and avoiding sudden temperature changes can help manage symptoms. "
                  "Over-the-counter anti-itch creams or prescribed topical corticosteroids may be recommended.<br><br>"
                  "<strong>When to Seek Medical Help:</strong> If eczema causes discomfort, disrupts sleep, or shows signs of infection, "
                  "consulting a dermatologist for targeted treatment is beneficial.<br><br>"
                  f"<a href='{disease_links['Eczema']}'>Learn more</a>")

    elif predicted_class == "RingWorm":
        output = (f"<h4>Ringworm</h4>"
                  "It’s typically treatable at home with antifungal creams or powders. "
                  "However, if the infection is widespread or doesn't improve after treatment, "
                  "a doctor’s consultation is needed, as prescription medication may be required.<br><br>"
                  "<strong>Causes:</strong> Ringworm is caused by a fungal infection and is often spread through skin-to-skin contact, "
                  "contact with contaminated surfaces, or exposure to infected animals.<br><br>"
                  "<strong>General Advice:</strong> Maintain good hygiene, avoid sharing personal items like towels, and keep the affected area clean and dry. "
                  "Over-the-counter antifungal creams are often effective for mild cases.<br><br>"
                  "<strong>When to Seek Medical Help:</strong> Seek medical advice if symptoms persist beyond two weeks, if the rash spreads or worsens, "
                  "or if it affects the scalp or other sensitive areas.<br><br>"
                  "<strong>Symptoms :</strong> of ringworm on your body usually start about 4 to 14 daysTrusted Source after contact with the fungus.<br>"""
                  " => Ringworm can affect any area of your skin, and it may also be found on fingernails and toenails.<br>"
                  "<strong>Symptoms usually include:<br> . a ring-shaped rash<br>. red skin that is scaly or cracked<br>. hair loss<br>. itchy skin</strong><br><br>"
                  "<strong>Causes and risk factors :</strong><br> =>Factors that may increase your risk include:<br> .living in damp, hot, or humid areas<br> .excessive sweating<br> .participating in contact sports<br> .wearing tight clothing<br> .having a weak immune system<br> .sharing clothing, bedding, or towels with others<br> .diabetes<br><br>"
                f"<a href='{disease_links['RingWorm']}'>Learn more</a>")
    elif predicted_class == "non-rash":
        output = "Unknown rash type."

    return output

# Create 'upload' directory if it doesn't exist
upload_directory = 'upload'
if not os.path.exists(upload_directory):
    os.makedirs(upload_directory)

# File uploader with customized styling
st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
uploadFile = st.file_uploader('Upload an Image', type=['jpg', 'png', 'jpeg'])
st.markdown('</div>', unsafe_allow_html=True)

if uploadFile is not None:
    # Save the uploaded file
    with open(os.path.join(upload_directory, uploadFile.name), 'wb') as f:
        f.write(uploadFile.getbuffer())

    # Classify the uploaded image
    image_path = os.path.join(upload_directory, uploadFile.name)
    result = classifyImages(image_path)

    st.image(uploadFile, width=200)
    # Display the result
    st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)
