# opensfm_api.py
import os
import subprocess
import traceback
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

opensfm_api = Blueprint('opensfm_api', __name__)

UPLOAD_BASE_PATH = "/home/ubuntu/OpenSfM/data"
RUN_SCRIPT_PATH = "/home/ubuntu/backend/OpenSfM/run.sh"
CALL_SCRIPT_PATH = "/home/ubuntu/backend/OpenSfM/data/0718_CallFilename.py"
SCREENSHOT_FOLDER_NAME = "screenshot"

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@opensfm_api.route('/images', methods=['POST'])
def handle_images():
    try:
        if 'userid' not in request.form:
            return jsonify({"error": "Missing userid"}), 400

        user_id = secure_filename(request.form['userid'])
        if 'images' not in request.files:
            return jsonify({"error": "Missing images"}), 400

        images = request.files.getlist('images')
        if not images or not any(allowed_file(img.filename) for img in images):
            return jsonify({"error": "No valid image files provided"}), 400

        user_dir = os.path.join(UPLOAD_BASE_PATH, user_id)
        os.makedirs(user_dir, exist_ok=True)

        for img in images:
            if img and allowed_file(img.filename):
                filename = secure_filename(img.filename)
                img.save(os.path.join(user_dir, filename))

        # Run run.sh
        result = subprocess.run(
            ["bash", "-c", f"cd /home/ubuntu/backend/OpenSfM && ./run.sh {user_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            return jsonify({
                "error": "run.sh execution failed",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500

        ply_path = os.path.join(user_dir, "undistorted", "depthmaps", "merged.ply")
        if not os.path.exists(ply_path):
            return jsonify({"error": "PLY output file not found"}), 500

        # Run 0718_CallFilename.py
        call_result = subprocess.run(
            [
                "bash", "-c",
                f"cd /home/ubuntu/backend/OpenSfM && "
                f"source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && "
                f"conda activate openSFM && "
                f"python data/0718_CallFilename.py {ply_path}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if call_result.returncode != 0:
            return jsonify({
                "error": "0718_CallFilename.py execution failed",
                "stdout": call_result.stdout,
                "stderr": call_result.stderr
            }), 500

        screenshot_dir = os.path.join(user_dir, SCREENSHOT_FOLDER_NAME)
        if not os.path.exists(screenshot_dir):
            return jsonify({"error": "Screenshot folder not found"}), 500

        image_files = sorted([
            f for f in os.listdir(screenshot_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(image_files) < 6:
            return jsonify({"error": f"Expected at least 6 screenshots, found {len(image_files)}"}), 500

        image_urls = [f"/screenshot/{user_id}/{f}" for f in image_files[:6]]
        messages = call_result.stdout.strip().splitlines()[2:4]
        assessment = call_result.stdout.strip().splitlines()[4:5]

        return jsonify({
            "messages": messages,
            "assessment": assessment,
            "images": image_urls
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@opensfm_api.route('/screenshot/<userid>/<filename>')
def serve_screenshot(userid, filename):
    safe_userid = secure_filename(userid)
    safe_filename = secure_filename(filename)
    directory = os.path.join(UPLOAD_BASE_PATH, safe_userid, SCREENSHOT_FOLDER_NAME)
    return send_from_directory(directory, safe_filename)
