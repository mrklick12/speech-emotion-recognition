from flask import Flask, render_template, request, redirect, url_for, flash, abort
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from pathlib import Path
import shutil
import os
from helper import predict_with_joblib # this is from helper.py


app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'wav'}

# Path object for filesystem operations
UPLOADS_PATH = Path(app.config['UPLOAD_FOLDER'])

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def is_file_allowed(filename: str) -> bool:
    if '.' in filename:
        if filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS: # checks in the filename is valid
            return True
    return False


@app.route("/")
def index():
    message = request.args.get('message', '')
    result = request.args.get('result', '') 
    # Requests for any input text from URL 

    return render_template("index.html", message=message, result=result)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: # checks if this file is coming from the upload form
        return redirect(url_for('index', message="No valid file."))

    file = request.files['file'] # stores the uploaded .wav in file

    if file.filename == '': # checks if empty
        return redirect(url_for('index', message="No selected file"))

    if not is_file_allowed(file.filename): 
        return redirect(url_for('index', message="Only .wav files are allowed"))


@app.route('/check', methods=['POST'])
def check_emotion():
    # DOES ALL THE CHECKS IN THE UPLOAD FUNCTION
    if 'file' not in request.files: 
        return redirect(url_for('index', message="No valid file."))

    file = request.files['file'] 

    if file.filename == '':
        return redirect(url_for('index', message="No selected file"))

    if not is_file_allowed(file.filename):
        return redirect(url_for('index', message="Only .wav files are allowed"))
        
    # Delete any files in the uploads folder so only ONE file is present at a time to reduce storage use
    folder = UPLOADS_PATH
    if folder.exists():
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    filename = secure_filename(file.filename) # sanitizes filename

    dest = os.path.join(app.config['UPLOAD_FOLDER'], filename) # creates destination path
    try:
        file.save(dest)
    except Exception as e:
        return redirect(url_for('index', message=f"Failed to save file: {e}"))

    # Model file sits in the package root (SVCaudioidentifier.joblib)
    model_path = os.path.join(app.root_path, 'SVCaudioidentifier.joblib')
    try:

        result = predict_with_joblib(model_path, dest) # this function is from helper.py, the model returns a string of outputs
        prediction = result.get('prediction')
        

        if hasattr(prediction, '__len__'): # checks if the prediction is an array or list essentially
            display = str(prediction[0]) # takes the actual prediction value
        else:
            display = str(prediction) # if its not a list (idk why it wouldn't but) then it just takes the value directly
        return redirect(url_for('index', result=display)) # DISPLAYS THE PREDICTION ON THE WEBPAGE
    

    except Exception as e:
        return redirect(url_for('index', message=f"Prediction error: {e}"))


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return redirect(url_for('index', message="File too large. 10MB limit."))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
