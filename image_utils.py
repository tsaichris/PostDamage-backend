from PIL import Image

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return f"Width: {width}px, Height: {height}px"

# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace this with the actual path to your image
    print(get_image_size(image_path))