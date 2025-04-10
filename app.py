from flask import Flask, render_template, request, send_file, url_for
import os
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('denoising_autoencoder.h5')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
DOWNLOAD_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def process_audio(audio_path):
    # Load and process audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Reshape audio to match model's expected input shape
    target_length = 1024 * 44
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    audio_processed = audio.reshape(1, 1024, 44, 1)
    denoised_audio = model.predict(audio_processed)
    denoised_audio = denoised_audio.reshape(-1)
    
    return denoised_audio, sr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/denoise', methods=['POST'])
def denoise():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    # Save uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)
    
    # Process audio
    denoised_audio, sr = process_audio(input_path)
    
    # Save processed audio
    output_filename = f'denoised_{file.filename}'
    output_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    sf.write(output_path, denoised_audio, sr)
    
    # Return the download URL
    return {'download_url': url_for('download_file', filename=output_filename)}

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(DOWNLOAD_FOLDER, filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True)