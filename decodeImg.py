import os
import gridfs
from pymongo import MongoClient
from PIL import Image
import io
import traceback

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017")  # Adjust the MongoDB URI as needed

output_folder = "decoded_images_0225"
os.makedirs(output_folder, exist_ok=True)
# List of collections to process in each database
collection_names = ['wall', 'beam', 'column']

def save_image_from_gridfs(fs, image_id, filename):
    """Retrieve and save an image from GridFS by its ID."""
    try:
        # Fetch the image from GridFS
        image_data = fs.get(image_id).read()
        image = Image.open(io.BytesIO(image_data))

        # Save the image to the output folder
        image.save(os.path.join(output_folder, filename))
        print(f"Image saved: {filename}")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError encountered for image {filename}: {e}")
        print(traceback.format_exc())
    except Exception as e:
        print(f"Failed to save image {filename}: {e}")
        print(traceback.format_exc())

# Iterate over all databases
for db_name in client.list_database_names():
    db = client[db_name]
    fs = gridfs.GridFS(db)

    # Iterate over each collection in the current database and process images
    for collection_name in collection_names:
        if collection_name in db.list_collection_names():
            collection = db[collection_name]
            documents = collection.find()

            for doc in documents:
                # Retrieve and save the 'original_img_id' image
                if "original_img_id" in doc:
                    original_img_id = doc["original_img_id"]
                    try:
                        save_image_from_gridfs(fs, original_img_id, f"{db_name}_{collection_name}_original_{original_img_id}.jpg")
                    except Exception as e:
                        print(f"Error processing original_img_id {original_img_id} in {db_name}.{collection_name}: {e}")

                # Retrieve and save the 'detected_img_id' image
                if "detected_img_id" in doc:
                    detected_img_id = doc["detected_img_id"]
                    try:
                        save_image_from_gridfs(fs, detected_img_id, f"{db_name}_{collection_name}_detected_{detected_img_id}.jpg")
                    except Exception as e:
                        print(f"Error processing detected_img_id {detected_img_id} in {db_name}.{collection_name}: {e}")

print("All images have been processed.")
