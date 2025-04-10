from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'denoising_autoencoder.h5'

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please ensure the model file is in the project root directory.")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def process_audio(audio_path):
    # Load and process audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Reshape audio to match model's expected input shape
    # Pad or truncate audio to fit 1024 * 44 samples
    target_length = 1024 * 44
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    # Reshape to (batch_size, height, width, channels)
    audio_processed = audio.reshape(1, 1024, 44, 1)
    
    # Make prediction
    denoised_audio = model.predict(audio_processed)
    
    # Reshape back to 1D array
    denoised_audio = denoised_audio.reshape(-1)
    
    return denoised_audio, sr

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/denoise', methods=['POST'])
def denoise():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'denoised_' + filename)
        
        # Save uploaded file
        file.save(input_path)
        
        # Process audio
        denoised_audio, sr = process_audio(input_path)
        
        # Save denoised audio
        sf.write(output_path, denoised_audio, sr)
        
        # Return the denoised file
        return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)