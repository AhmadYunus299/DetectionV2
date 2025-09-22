from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = tf.keras.models.load_model('model_penyakit_tanaman_cabai.h5')

def predict_disease(chili_plant):
    try:
        test_image = load_img(chili_plant, target_size=(100, 100))
        test_image = img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        pred = np.argmax(result, axis=1)[0]

        labels = [
            "Chili - Antracnose",
            "Chili - Healthy",
            "Chili - Leaf Spot",
            "Chili - Leaf Crul",
            "Chili - Whitely",
            "Chili - Yellowwish",
            "Chili - Dumping Off"
        ]
        
        if pred < len(labels):
            return labels[pred], f'{labels[pred].lower().replace(" ", "_")}.html'
        else:
            return "Penyakit tidak terdeteksi", 'unknown.html'
    except Exception as e:
        return "Penyakit tidak terdeteksi", 'unknown.html'

def predict_frame(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (100, 100))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)
        result = model.predict(input_frame)
        pred = np.argmax(result, axis=1)[0]
        confidence = np.max(result)
        labels = ["Chili - Antracnose", "Chili - Healthy", "Chili - Leaf Spot", "Chili - Leaf Crul", "Chili - Whitely", "Chili - Yellowwish", "Chili - Dumping Off"]
        return labels[pred], confidence
    except Exception as e:
        return "Penyakit tidak terdeteksi", 0

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('dashboard.html', active_menu='')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        pred, output_page = predict_disease(chili_plant=file_path)
        return render_template(output_page, pred_output=pred, user_image=file_path)
    return render_template('clasification.html', active_menu='predict')

@app.route("/history", methods=['GET', 'POST'])
def history():
    return render_template('history.html', active_menu='history')

camera_active = False

def gen_frames():
    global camera_active
    camera = cv2.VideoCapture(0)
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        else:
            pred, confidence = predict_frame(frame)
            label = f"{pred} - {confidence:.2f}"
            
            # Drawing a bounding box if a disease is detected
            if pred not in ["Chili - Healthy"]:
                height, width, _ = frame.shape
                start_point = (int(width * 0.2), int(height * 0.2))
                end_point = (int(width * 0.8), int(height * 0.8))
                color = (0, 255, 0)
                thickness = 2
                frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            
            # Adding label
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global camera_active
    camera_active = False
    return jsonify(result="Camera stopped")

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
