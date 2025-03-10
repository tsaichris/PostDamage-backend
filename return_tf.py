from PIL import Image
from flask import jsonify

def get_image_string(image_path):
    # Open the image and retrieve the width as an integer
    with Image.open(image_path) as img:
        width = img.width
    return width  # Return the width directly as an integer

def toggle_definition(image_width):
    # Check if the width is greater than 1500
    if image_width > 1000:
        return True
    else:
        # Return an error message indicating the photo quality isn't enough
        return jsonify({"error": "Photo quality isn't enough"}), 410
