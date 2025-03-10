from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from utli import DamageClassification, ImageQuality, DamageDetection, checkReason
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
from gevent import monkey
import numpy as np
from bson import Binary
import gridfs
from io import BytesIO
from bson.objectid import ObjectId
import io
import random
import string
import copy
import ssl
import pickle

monkey.patch_all()

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
EXPORT_FOLDER = './exports'
EXCEL_EXPORT_PATH = os.path.join(EXPORT_FOLDER, 'test_database.xlsx')
IMAGE_DOWNLOAD_PATH = './images'  # Folder to temporarily store images

app = Flask(__name__)

# Load SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS)
context.load_cert_chain(certfile="/home/ubuntu/backend/cert.pem", keyfile="/home/ubuntu/backend/key.pem")

app.logger.setLevel(logging.INFO)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, resources={
    r"/*": {
        "origins": "*",  # In production, replace with your specific domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

client = MongoClient('mongodb://localhost:27017/')


# Pre-defined database and collections
load_history_db = client['load_history']
predefined_collections = ['class_01', 'class_02', 'class_03']
for collection in predefined_collections:
    if collection not in load_history_db.list_collection_names():
        load_history_db.create_collection(collection)
        app.logger.info(f"Created collection: {collection} in database: load_history")

# Initialize the test-database for other use cases, but we will change it dynamically based on userId
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
        model_path = f'Models/damageClassify/{detectionType}/'  # Adjust this path as necessary
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
        ####################### change path ##########################
        model_path = f'Models/crackClassify/{detectionType}/'  # Adjust this path as necessary
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


def generate_random_id(length=16):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


@app.route("/get_user_id", methods=['GET'])
def get_user_id():
    """Return the newly generated user ID."""
    user_id = str(generate_random_id())
    return jsonify({"userId": user_id}), 200

# Global variables for the database
db = None
fs = None 

@app.route("/image", methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        try:
            app.logger.info("Received POST request to /image")
            app.logger.info(f"Request headers: {request.headers}")
            app.logger.info(f"Request files: {request.files}")
            app.logger.info(f"Request form: {request.form}")

            # CORS headers for no-cors mode
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Content-Type': 'application/octet-stream'  # Changed to binary data type
            }
            
            if 'file' not in request.files:
                return Response(
                    pickle.dumps({"success": False, "error": "No file part"}),
                    status=400,
                    headers=headers
                )
            
            file = request.files['file']
            if file.filename == '':
                return Response(
                    pickle.dumps({"success": False, "error": "No selected file"}),
                    status=401,
                    headers=headers
                )
            
            user_id = request.form.get('userId')
            detection_type = request.form.get('detectionType')


            if not user_id or not detection_type:
                return Response(
                    pickle.dumps({"success": False, "error": "Missing userId or detectionType"}),
                    status=400,
                    headers=headers
                )

            # Create a new database and GridFS based on userId
            db = client[user_id]
            fs = gridfs.GridFS(db)

            # Ensure the three required collections are created
            collection_names = ['column', 'beam', 'wall']
            for collection in collection_names:
                if collection not in db.list_collection_names():
                    db.create_collection(collection)
                    app.logger.info(f"Created collection: {collection} in database: {user_id}")

            # Access the relevant collection based on detection type
            #collection_map = {
            #    'column': db['column'],
             #   'beam': db['beam'],
              #  'wall': db['wall']
            #}
            collection = db[detection_type]
            #collection = collection_map.get(detection_type)

            if collection is None:
                return Response(
                    pickle.dumps({"success": False, "error": f"Invalid detection type: {detection_type}"}),
                    status=400,
                    headers=headers
                )

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Read the file as an image for processing using OpenCV and PIL
                image_cv = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                image_PIL = PILImage.open(request.files['file']).convert("RGB")
                
                # Perform image quality check
                app.logger.info("Performing image quality check")
                image_result = ImageQuality(image_cv)
                if not image_result:
                    return Response(
                        pickle.dumps({"success": False, "error": "Image quality insufficient"}),
                        status=420,
                        headers=headers
                    )
                if detection_type not in models_classification or models_classification[detection_type] is None:
                    return Response(
                        pickle.dumps({"success": False, "error": f"Model for {detection_type} not loaded"}),
                        status=500,
                        headers=headers
                    )
                
                app.logger.info("Performing damage classification")
                
                classify_result = DamageClassification(image_PIL, models_classification[detection_type])
                app.logger.info("Performing damage detection")
                total_result, original_img, detected_img = DamageDetection(image_PIL,image_cv, model_yolo, models_classification_crack[detection_type], detection_type)
                app.logger.info("Performing Reason check")
                classType, reason = checkReason(detection_type,classify_result,total_result)
                app.logger.info("Model complete")

                # Store the original image in GridFS (unchanged)
                file.seek(0)  # Reset file pointer to the start for GridFS storage
                original_img_id = fs.put(file, filename=filename)  # Store original image in GridFS

                # Resize the detected image to 640x640
                app.logger.info("Resizing detected image to 640x640")
                
                detected_img_resized = detected_img.resize((640, 640))

                # Convert the resized image to bytes for storage in GridFS
                detected_img_bytes = io.BytesIO()
                detected_img_resized.save(detected_img_bytes, format='JPEG')
                detected_img_bytes.seek(0)  # Reset buffer pointer to the start for reading

                # Store the resized detected image in GridFS
                detected_img_id = fs.put(detected_img_bytes, filename=f"detected_{filename}")

                app.logger.info("Classification complete, storing result in database")

                # Store metadata and image references in the collection
                collection.insert_one({
                    "original_img_id": original_img_id,  # Store GridFS ID for original image
                    "detected_img_id": detected_img_id,  # Store GridFS ID for detected image
                    "reason": reason,
                    "resultlist": classType,
                    "detectionType": detection_type
                })

                # Prepare response data
                response_data = {
                    "success": True,
                    "message": "Image processed successfully",
                    "data": {
                        "classType": classType,
                        "reason": reason,
                        "originalImageId": str(original_img_id),
                        "detectedImageId": str(detected_img_id),
                        "classificationResult": classify_result,
                        "detectionResult": total_result,
                        #"detectedImage": detected_img_base64  # Include base64 image in response
                    }
                }
                
                # Convert response to binary
                return Response(
                    pickle.dumps(response_data),
                    status=200,
                    headers=headers
                )

            else:
                return Response(
                    pickle.dumps({"success": False, "error": "File type not allowed"}),
                    status=402,
                    headers=headers
                )

        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return Response(
                pickle.dumps({
                    "success": False,
                    "error": "Internal server error",
                    "details": str(e)
                }),
                status=500,
                headers=headers
            )

    elif request.method == 'OPTIONS':
        return Response(
            pickle.dumps({"success": True, "message": "OK"}),
            status=200,
            headers=headers
        )

    return Response(
        pickle.dumps({"success": False, "error": "Method not allowed"}),
        status=405,
        headers=headers
    )



    
from flask import Flask, jsonify, request, send_file
import os
import base64
import gridfs
from bson import ObjectId

# app = Flask(__name__)

BASE_PATH = "/home/yuio7012/app_temp/backend/resultpage"

@app.route("/results/check_overall/<string:userid>", methods=['POST', 'GET'])
def check_overall(userid):
    db_name = userid
    app.logger.info(f"Received userid (db_name): {db_name}")

    # Check if user wants to fetch the image directly
    if request.args.get('image'):
        image_path = request.args.get('image')
        # Serve the image from the filesystem (local path)
        return serve_image_overall(image_path)

    # Access the specified database
    try:
        db = client[db_name]
    except Exception as e:
        return jsonify({"error": f"Failed to access database: {str(e)}"}), 500

    try:
        seriousness = {"Class A": 1, "Class B": 2, "Class C": 3}
        overall_seriousness = float('inf')

        def get_image_path(classification):
            # Construct image paths for each classification from the filesystem
            if classification == "Class A":
                return os.path.join(BASE_PATH, "Level_A.png")
            elif classification == "Class B":
                return os.path.join(BASE_PATH, "Level_B.png")
            elif classification == "Class C":
                return os.path.join(BASE_PATH, "Level_C.png")
            else:
                return None  # Return None for invalid classification

        # Iterate over collections and determine the overall classification
        for collection_name in ['column', 'beam', 'wall']:
            collection = db[collection_name]
            documents = list(collection.find())

            current_seriousness_level = float('inf')
            for doc in documents:
                if "resultlist" in doc:
                    resultlist = doc["resultlist"]
                    if isinstance(resultlist, str):
                        if "Class A" in resultlist and seriousness["Class A"] < current_seriousness_level:
                            current_seriousness_level = seriousness["Class A"]
                        elif "Class B" in resultlist and seriousness["Class B"] < current_seriousness_level:
                            current_seriousness_level = seriousness["Class B"]
                        elif "Class C" in resultlist and seriousness["Class C"] < current_seriousness_level:
                            current_seriousness_level = seriousness["Class C"]

            if current_seriousness_level < overall_seriousness:
                overall_seriousness = current_seriousness_level

        # Determine the overall classification based on the lowest seriousness level found
        if overall_seriousness == seriousness["Class A"]:
            overall_classification = "Class A"
        elif overall_seriousness == seriousness["Class B"]:
            overall_classification = "Class B"
        elif overall_seriousness == seriousness["Class C"]:
            overall_classification = "Class C"
        else:
            overall_classification = "No classification"

        overall_image_path = get_image_path(overall_classification)

        if overall_image_path and os.path.exists(overall_image_path):
            # Serve image from filesystem
            image_url = f"/results/check_overall/{userid}?image={overall_image_path}"
            image_data = serve_image_as_base64_overall(overall_image_path)
            image_message = "Image successfully sent to front-end." if image_data else "Failed to send image to front-end."
        else:
            image_url = None
            image_data = None
            image_message = "No image available."

        output = {
            "overall_classification": overall_classification,
            "image_url": image_url if image_url else "No image available",  # Send URL or a message
            "image_data": image_data if image_data else "No image data available",  # Send base64 image data
            "image_message": image_message  # Message indicating success or failure of image sending
        }

        return jsonify(output), 200

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


def serve_image_as_base64_overall(image_path):
    """Helper function to serve an image from the filesystem as base64-encoded binary for embedding in JSON."""
    if image_path:
        if os.path.exists(image_path):
            try:
                # Open the image and encode it as base64
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                return encoded_string
            except Exception as e:
                app.logger.error(f"Error serving image as base64: {str(e)}")
                return None
        else:
            app.logger.warning(f"Image path does not exist: {image_path}")
            return None
    else:
        app.logger.warning("No image path provided.")
        return None


def serve_image_overall(image_path):
    """Helper function to serve an image from the filesystem."""
    if image_path:
        if os.path.exists(image_path):
            try:
                # Open the image and encode it as base64
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                return jsonify({"image_data": encoded_string}), 200
            except Exception as e:
                app.logger.error(f"Error serving image: {str(e)}")
                return jsonify({"error": f"Failed to serve image: {str(e)}"}), 500
        else:
            app.logger.warning(f"Image path does not exist: {image_path}")
            return jsonify({"error": "Image not found"}), 404
    else:
        app.logger.warning("No image path provided.")
        return jsonify({"error": "Image path is missing"}), 400


from flask import Flask, jsonify, request
import base64
from bson import ObjectId
import gridfs

# Global variables for the database
db = None

@app.before_request
def before_request_func():
    global db
    userid = request.view_args.get('userid') if request.view_args else None
    if userid:
        db = client[userid]  # Replace 'client' with actual MongoDB client connection
        
def serve_image_overall(image_path):
    """Helper function to serve an image from the filesystem."""
    if image_path:
        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                return jsonify({"image_data": encoded_string}), 200
            except Exception as e:
                app.logger.error(f"Error serving image: {str(e)}")
                return jsonify({"error": f"Failed to serve image: {str(e)}"}), 500
        else:
            app.logger.warning(f"Image path does not exist: {image_path}")
            return jsonify({"error": "Image not found"}), 404
    else:
        app.logger.warning("No image path provided.")
        return jsonify({"error": "Image path is missing"}), 400


from flask import Flask, jsonify, request
import base64
from bson import ObjectId
import gridfs

# Global variables for the database
db = None

@app.before_request
def before_request_func():
    global db
    userid = request.view_args.get('userid') if request.view_args else None
    if userid:
        db = client[userid]  # Replace 'client' with actual MongoDB client connection
        
@app.before_request
def before_request():
    if request.is_secure:
        return
    url = request.url.replace("http://", "https://", 1)
    return redirect(url, code=301)

@app.before_request
def force_https():
    if not request.is_secure and app.env != 'development':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)
        
@app.route("/results/check_detailed/<string:userid>", methods=['POST', 'GET'])
def check_detailed_results(userid):
    global db  # Access the global db variable
    
    app.logger.info(f"Received userid (db_name): {userid}")

    try:
        seriousness = {"Class A": 1, "Class B": 2, "Class C": 3}
        final_results = {}
        fs = gridfs.GridFS(db)  # Initialize GridFS for this user

        # Iterate over all collections: column, beam, and wall
        for collection_name in ['column', 'beam', 'wall']:
            collection = db[collection_name]
            documents = list(collection.find())

            most_serious_result = None
            current_seriousness_level = float('inf')

            # Iterate through the documents to find the most serious result in this collection
            for doc in documents:
                if "resultlist" in doc:
                    resultlist = doc["resultlist"]
                    if isinstance(resultlist, str):
                        # Determine seriousness based on Class A, B, C
                        if "Class A" in resultlist and seriousness["Class A"] < current_seriousness_level:
                            most_serious_result = doc
                            current_seriousness_level = seriousness["Class A"]
                        elif "Class B" in resultlist and seriousness["Class B"] < current_seriousness_level:
                            most_serious_result = doc
                            current_seriousness_level = seriousness["Class B"]
                        elif "Class C" in resultlist and seriousness["Class C"] < current_seriousness_level:
                            most_serious_result = doc
                            current_seriousness_level = seriousness["Class C"]

             # If the most serious result is found in the collection, process it
            if most_serious_result:
                reason = most_serious_result.get("reason", "No reason provided")

                # Retrieve the original and detected image from GridFS
                original_img_id = most_serious_result.get("original_img_id")
                detected_img_id = most_serious_result.get("detected_img_id")

                # Initialize variables for the images
                original_img_base64 = "No original image data available"
                detected_img_base64 = "No detected image data available"

                # Retrieve original image from GridFS
                if original_img_id:
                    original_img_data = fs.get(ObjectId(original_img_id)).read()
                    original_img_base64 = base64.b64encode(original_img_data).decode('utf-8')
                    app.logger.info(f"Original image data successfully retrieved and encoded for {collection_name}")
                else:
                    app.logger.warning(f"No original image found for {collection_name}")

                # Retrieve detected image from GridFS
                if detected_img_id:
                    detected_img_data = fs.get(ObjectId(detected_img_id)).read()
                    detected_img_base64 = base64.b64encode(detected_img_data).decode('utf-8')
                    app.logger.info(f"Detected image data successfully retrieved and encoded for {collection_name}")
                else:
                    app.logger.warning(f"No detected image found for {collection_name}")

                # Prepare the output for this collection
                final_results[collection_name] = {
                    "collection_name": collection_name,
                    "resultlist": most_serious_result.get("resultlist", "No resultlist"),
                    "reason": reason,
                    "detected_image_base64": detected_img_base64
                }

        # If any collection had serious results, return them
        if final_results:
            return jsonify(final_results), 200
        else:
            return jsonify({"message": "No serious results found in any collection"}), 200

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

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
# for ngrok page    
@app.route('/')
def home():
    return "Welcome to the backend server!"

from gevent.pywsgi import WSGIServer
from gevent.ssl import SSLContext
if __name__ == '__main__':


    # SSL context configuration
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain(
            certfile='/home/ubuntu/backend/cert.pem',
            keyfile='/home/ubuntu/backend/key.pem'
        )
    except Exception as e:
        app.logger.error(f"Failed to load SSL certificates: {e}")
        raise

    # Configure SSL-enabled server
    ssl_server = WSGIServer(
        ('0.0.0.0', 8443),
        application=app,
        ssl_context=context,
        log=app.logger
    )

    app.logger.info("Starting secure server on port 443...")
    try:
        ssl_server.serve_forever()
    except Exception as e:
        app.logger.error(f"Server failed to start: {e}")
        raise