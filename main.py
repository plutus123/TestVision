from flask import Flask, request, jsonify, render_template, send_file
from model import process_image_and_generate_test_cases
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import base64
from io import BytesIO

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_test_cases', methods=['POST'])
def generate_test_cases():
    data = request.json
    context = data.get('context', '')
    images = data.get('images', [])

    try:
        decoded_images = []
        for image_base64 in images:
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            decoded_images.append(image)

        test_cases, annotated_images = process_image_and_generate_test_cases(decoded_images, context)

        annotated_images_base64 = []
        for img in annotated_images:
            _, buffer = cv2.imencode('.png', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            annotated_images_base64.append(img_base64)

        return jsonify({
            "test_cases": test_cases,
            "annotated_images": annotated_images_base64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)