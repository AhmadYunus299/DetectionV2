from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_mysqldb import MySQL
import tensorflow as tf
import numpy as np
import os
import cv2
from collections import defaultdict
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'deteksi_penyakit'
mysql = MySQL(app)

model = tf.keras.models.load_model('model_penyakit_tanaman_cabai.h5')

def predict_disease(chili_plant):
    test_image = load_img(chili_plant, target_size=(100, 100))
    test_image = img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    pred = np.argmax(result, axis=1)
    if pred == 0:
        return "Chili - Antracnose", 'antracnose.html'
    elif pred == 1:
        return "Chili - Healthy", 'healty.html'
    elif pred == 2:
        return "Chili - Leaf Spot", 'leaf_spot.html'
    elif pred == 3:
        return "Chili - Leaf Crul", 'leaf_crul.html'
    elif pred == 4:
        return "Chili - Whitely", 'whitely.html'
    elif pred == 5:
        return "Chili - Yellowwish", 'yellowwish.html'
    elif pred == 6:
        return "Chili - Dumping Off", 'dumping_of.html'
    return None, None

def predict_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (100, 100))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    result = model.predict(input_frame)
    pred = np.argmax(result, axis=1)[0]
    confidence = np.max(result)
    labels = ["Chili - Antracnose", "Chili - Healthy", "Chili - Leaf Spot", "Chili - Leaf Crul", "Chili - Whitely", "Chili - Yellowwish", "Chili - Dumping Off"]
    return labels[pred], confidence

@app.route("/", methods=['GET', 'POST'])
def main():
# Create a cursor
    cur = mysql.connection.cursor()
    
    # Query to fetch data
    query = """
    SELECT id, nama, tanggal, waktu
    FROM riwayat
    """
    cur.execute(query)
    
    # Fetch all records
    records = cur.fetchall()
    
    # Close the cursor
    cur.close()
    
    # Convert records to a list of dictionaries
    result = []
    for row in records:
        result.append({
            "id": row[0],
            "nama": row[1],
            "tanggal": row[2],
            "waktu": row[3]
        })
    
    return render_template('dashboard.html', data=result)

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

@app.route("/history", methods=['GET'])
def history():
    page = int(request.args.get('page', 1))
    per_page = 10  # Fixed number of items per page

    cursor = mysql.connection.cursor()
    offset = (page - 1) * per_page

    cursor.execute("SELECT COUNT(*) FROM riwayat")
    total_items = cursor.fetchone()[0]
    
    cursor.execute("SELECT nama, tanggal, waktu, gambar FROM riwayat LIMIT %s OFFSET %s", (per_page, offset))
    data = cursor.fetchall()
    cursor.close()

    history_data = [{
        'nama': row[0],
        'tanggal': row[1],
        'waktu': row[2],
        'gambar': row[3]
    } for row in data]

    total_pages = (total_items + per_page - 1) // per_page

    return render_template(
        'history.html',
        active_menu='history',
        history_data=history_data,
        current_page=page,
        total_pages=total_pages
    )
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
            # Example condition for demonstration; replace with actual condition
            if pred in ["Chili - Antracnose", "Chili - Leaf Spot"]:  # Modify this condition based on detected diseases
                height, width, _ = frame.shape
                start_point = (int(width * 0.2), int(height * 0.2))  # Top-left corner
                end_point = (int(width * 0.8), int(height * 0.8))    # Bottom-right corner
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

@app.route('/save', methods=['POST'])
def save():
    if request.method == 'POST':
        # file = request.files['image']
        # filename = file.filename
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)
        # pred, output_page = predict_disease(chili_plant=file_path)

        nama = request.form['name']
        tanggal = request.form['tanggal']
        waktu = request.form['waktu']
        gambar = request.form['gambar']

        print(nama)
        print(tanggal)
        print(waktu)
        print(gambar)

        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO riwayat (nama, tanggal, waktu, gambar) VALUES(%s, %s, %s, %s)''', (nama, tanggal, waktu, gambar))
        mysql.connection.commit()
        cursor.close()

        return redirect('/history')
        # return render_template('clasification.html', active_menu='predict')

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
