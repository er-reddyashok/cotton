# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Create Flask instance
app = Flask(__name__)

# Ensure the directory for user-uploaded images exists
UPLOAD_FOLDER = 'static/user_uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
try:
    model = load_model("model/v4_1_pred_cott_dis (1).h5")
    print('@@ Model loaded')
except Exception as e:
    print(f'@@ Error loading model: {e}')

def pred_cot_dieas(cott_plant):
    # Load image
    try:
        test_image = load_img(cott_plant, target_size=(150, 150))  # Load image with target size
        print("@@ Got Image for prediction")
    except Exception as e:
        print(f'@@ Error loading image: {e}')
        return "Error loading image. Ensure the image format is correct.", 'error.html'
    
    # Preprocess image
    try:
        test_image = img_to_array(test_image) / 255.0  # Convert image to numpy array and normalize
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    except Exception as e:
        print(f'@@ Error preprocessing image: {e}')
        return "Error preprocessing image", 'error.html'
    
    # Predict
    try:
        result = model.predict(test_image).round(3)  # Predict and round the result
        print('@@ Raw result = ', result)
        pred = np.argmax(result)  # Get the index of the max value
        print(f'@@ Predicted class index = {pred}')
    except Exception as e:
        print(f'@@ Error during prediction: {e}')
        return "Error during prediction", 'error.html'

    # Map prediction to output
    if pred == 0:
        return "diseased cotton leaf", 'disease_plant.html'
    elif pred == 1:
        return 'Diseased Cotton Plant', 'disease_plant.html'
    elif pred == 2:
        return 'fresh cotton leaf', 'healthy_plant_leaf.html'
    else:
        return "fresh cotton plant", 'healthy_plant.html'
# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Handle image upload, prediction, and render result page
@app.route("/predict", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('error.html', error_message="No file part in the request"), 400
    
    file = request.files['image']
    if file.filename == '':
        return render_template('error.html', error_message="No file selected"), 400

    if file:
        filename = file.filename
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)
            
        return render_template(output_page, pred_output=pred, user_image=file_path)
    else:
        return render_template('error.html', error_message="No file uploaded"), 400

# Run the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
