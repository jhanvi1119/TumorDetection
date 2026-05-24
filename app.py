from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from brestCancer import breast
from Brain_tumor import run

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploadBrain')
def uploadBrain():
    return render_template('uploadBrain.html')

@app.route('/uploadBreast')
def uploadBreast():
    return render_template('uploadBreast.html')

@app.route('/predictBrain', methods=['POST','GET'])
def predictBrain():
    if "image" not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files["image"]

    # Read the image file
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Process the image
    processed_image,result_str = run(image)

    # get url for input image
    _, original_img_encoded = cv2.imencode('.png', image)
    original_image_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

    
    # Convert output image to base64
    _, img_encoded = cv2.imencode('.png', processed_image)
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # return jsonify({'image_base64': image_base64})
    return render_template('predict.html',image_base64=image_base64, result_str=result_str, original_image_base64=original_image_base64)


@app.route('/predictBreast', methods=['POST','GET'])
def predictBreast():
    if "image" not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files["image"]

    # Read the image file
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Process the image
    processed_image,result_str = breast(image)

    # get url for input image
    _, original_img_encoded = cv2.imencode('.png', image)
    original_image_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

    
    # Convert output image to base64
    _, img_encoded = cv2.imencode('.png', processed_image)
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # return jsonify({'image_base64': image_base64})
    return render_template('predictBreast.html',image_base64=image_base64, result_str=result_str, original_image_base64=original_image_base64)


if __name__ == '__main__':
    app.run(debug=True, port=8000)