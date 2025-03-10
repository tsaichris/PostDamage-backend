from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
# from image_utils import get_image_size
# from return_tf import get_image_string, toggle_definition
from pymongo import MongoClient
from utli import DamageClassification,ImageQuality,DamageDetection,checkReason
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
import cv2
from PIL import Image as PILImage
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from logging.handlers import RotatingFileHandler
import logging
import uuid

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
EXPORT_FOLDER = './exports'
EXCEL_EXPORT_PATH = os.path.join(EXPORT_FOLDER, 'test_database.xlsx')
IMAGE_DOWNLOAD_PATH = './images'  # Folder to temporarily store images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

client = MongoClient('mongodb://localhost:27017/')
db = client['test-database']

# Set up logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)
app.logger.addHandler(file_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
console_handler.setLevel(logging.DEBUG)
app.logger.addHandler(console_handler)

app.logger.setLevel(logging.DEBUG)


models_classification = {
    'wall': None,
    'column': None,
    'beam': None
}

models_classification_crack = {
    'wall': None,
    'column': None,
    'beam': None
}


def load_model_damageClassification(detectionType):
    try:
        model_path = f'Models/old/{detectionType}/'  # Adjust this path as necessary
        app.logger.info(f"Attempting to load damage classification Model from {model_path}")
        model = ViTForImageClassification.from_pretrained(model_path)
        app.logger.info("damage classification Model loaded successfully")
        return model
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        #app.logger.error(traceback.format_exc())
        model = None

def load_model_crackClassification(detectionType):
    try:
        ####################### 路徑要改 ##########################
        model_path = f'Models/old/{detectionType}/'  # Adjust this path as necessary
        app.logger.info(f"Attempting to load crack classification Model from {model_path}")
        model = ViTForImageClassification.from_pretrained(model_path)
        app.logger.info("Crack classification Model loaded successfully")
        return model
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        #app.logger.error(traceback.format_exc())
        model = None

# Load the model when the app starts
models_classification['wall'] = load_model_damageClassification('wall')
models_classification['column'] = load_model_damageClassification('column')
models_classification['beam'] = load_model_damageClassification('beam')

models_classification_crack['wall'] = load_model_crackClassification('wall')
models_classification_crack['column'] = load_model_crackClassification('column')
models_classification_crack['beam'] = load_model_crackClassification('beam')


from ultralytics import YOLO
model_yolo = YOLO('Models/best.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/get_user_id", methods=['GET'])
def get_user_id():
    new_id = str(uuid.uuid4())
    return jsonify({"userId": new_id}), 200


@app.route("/image", methods=['POST', 'GET'])
def upload_file():
    global model
    if request.method == 'POST':
        try:
            app.logger.info("Received POST request to /image")
            app.logger.info(f"Request headers: {request.headers}")
            app.logger.info(f"Request files: {request.files}")
            app.logger.info(f"Request form: {request.form}")

            if 'file' not in request.files:
                app.logger.warning("No file part in the request")
                return jsonify({"error": "No file part"}), 400
            
            file = request.files['file']
            if file.filename == '':
                app.logger.warning("No selected file")
                return jsonify({"error": "No selected file"}), 401

            user_id = request.form.get('userId')
            detection_type = request.form.get('detectionType')

            if not user_id or not detection_type:
                app.logger.warning("Missing userId or detectionType")
                return jsonify({"error": "Missing userId or detectionType"}), 400

            collection_name = f"{user_id}-{detection_type}"
            collection = db[collection_name]
            app.logger.info(f"Collection name: {collection_name}")

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                app.logger.info(f"File saved to {filepath}")
                
                image_cv = cv2.imread(filepath)
                image_PIL = PILImage.open(filepath).convert("RGB")
                
                # Perform image quality check
                app.logger.info("Performing image quality check")
                image_result = ImageQuality(image_cv)
                if not image_result:
                    app.logger.warning("Image quality insufficient")
                    return jsonify({"Failed": "Image quality insufficient"}), 420

                if detection_type not in models_classification or models_classification[detection_type] is None:
                    app.logger.warning(f"Model for {detection_type} not loaded")
                    return jsonify({"error": f"Model for {detection_type} not loaded. Check server logs for details."}), 500
                
                app.logger.info("Performing damage classification")
                # if 'wall' in request.files: model = model_wall
                classify_result = DamageClassification(image_PIL, models_classification[detection_type])
                total_result = DamageDetection(image_PIL, model_yolo, models_classification_crack[detection_type], detection_type)
                classType, reason = checkReason(detection_type,classify_result,total_result)
                
                app.logger.info("Classification complete, returning result")
                
                collection.insert_one({
                "photourl": filepath,
                "resultlist": classify_result
                })
                
                return jsonify({
                    "detectionType":detection_type,
                    "classType": classType,
                    "reason": reason,
                    "message": "File uploaded successfully",
                }), 200
            else:
                app.logger.warning(f"File type not allowed: {file.filename}")
                return jsonify({"error": "File type not allowed"}), 402
        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
    elif request.method == 'GET':
        app.logger.info("Received GET request to /image")
        return jsonify({
            "message": "Use POST to upload an image to this endpoint."
        }), 200
    return jsonify({"error": "Method not allowed"}), 405
    
@app.route("/check_results", methods=['GET'])
def check_results():
    output = {}
    
    # Iterate through all collections in the MongoDB 'test-database'
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        # Fetch all documents from the collection
        documents = list(collection.find())

        result = "without input of this category or something went wrong"

        # Check for specific class strings in 'resultlist'
        for doc in documents:
            if "resultlist" in doc:
                resultlist = doc["resultlist"]
                
                # Since resultlist is a string, directly check for substrings
                if isinstance(resultlist, str):
                    if "Class A" in resultlist:
                        result = "Class A"
                        break
                    elif "Class B" in resultlist:
                        result = "Class B"
                        break
                    elif "Class C" in resultlist:
                        result = "Class C"
                        break
                else:
                    # Log or handle cases where resultlist is not a string
                    print(f"Skipping non-string resultlist: {resultlist}")

        # Add the collection result to the output
        output[collection_name] = result

    return jsonify(output), 200


@app.route("/export_excel", methods=['GET'])
def export_to_excel():
    # Ensure the export directory exists
    if not os.path.exists(EXPORT_FOLDER):
        os.makedirs(EXPORT_FOLDER)

    # Ensure the image download directory exists
    if not os.path.exists(IMAGE_DOWNLOAD_PATH):
        os.makedirs(IMAGE_DOWNLOAD_PATH)

    # Create an Excel writer using pandas
    with pd.ExcelWriter(EXCEL_EXPORT_PATH, engine='openpyxl') as writer:
        # Iterate through all collections in the MongoDB 'test-database'
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            # Fetch all documents from the collection
            documents = list(collection.find())
            # Convert documents to a pandas DataFrame
            df = pd.DataFrame(documents)

            # Write the DataFrame to a separate sheet in the Excel file
            df.to_excel(writer, sheet_name=collection_name, index=False)
    
    # Insert images into the generated Excel file
    workbook = load_workbook(EXCEL_EXPORT_PATH)

    for collection_name in db.list_collection_names():
        worksheet = workbook[collection_name]
        collection = db[collection_name]
        documents = list(collection.find())

        for row_num, document in enumerate(documents, start=2):  # start=2 to account for header
            if "photourl" in document:
                image_url = document["photourl"]

                try:
                    # If the URL is valid and accessible, download the image
                    image_path = download_image(image_url)
                    if image_path:
                        # Load the image and get its dimensions (width and height in pixels)
                        with PILImage.open(image_path) as img:
                            img_width, img_height = img.size

                        # Convert pixels to Excel column width and row height units
                        excel_width = img_width / 7.5  # Roughly 7.5 pixels per Excel unit width
                        excel_height = img_height * 0.75  # Roughly 0.75 pixels per Excel row height unit

                        # Adjust the Excel column width and row height to fit the image
                        worksheet.column_dimensions['B'].width = excel_width
                        worksheet.row_dimensions[row_num].height = excel_height

                        # Insert the image into the Excel cell
                        img = OpenpyxlImage(image_path)
                        img.anchor = f'B{row_num}'  # Insert image in column B (adjust as necessary)
                        worksheet.add_image(img)
                except Exception as e:
                    print(f"Failed to download or insert image: {e}")

    # Save the workbook with images
    workbook.save(EXCEL_EXPORT_PATH)

    # Return the Excel file as a download
    return send_file(EXCEL_EXPORT_PATH, as_attachment=True)

def download_image(image_url):
    """
    Download an image from a URL or file path and return the local path to the downloaded image.
    """
    if image_url.startswith('http'):
        try:
            # Fetch the image from the URL
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                # Extract image filename from the URL
                image_filename = os.path.basename(image_url)
                image_path = os.path.join(IMAGE_DOWNLOAD_PATH, image_filename)
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return image_path
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    elif os.path.exists(image_url):
        # If the image URL is a local file path, return it directly
        return image_url
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.debug = True
    # app.run()
