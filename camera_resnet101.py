from flask import Flask, render_template, request, Response, jsonify, flash, redirect, url_for
from flask_mysqldb import MySQL
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
import numpy as np
import cv2
import os

# ================= FLASK APP =================
app = Flask(__name__)
app.secret_key = "secretkey123"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================= MYSQL CONFIG =================
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'deteksi_penyakit'
mysql = MySQL(app)

# ================= MODEL =================
input_shape = (224, 224, 3)
num_classes = 4

inputs = Input(shape=input_shape, name="input_image")
base_model = ResNet101(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(name='gap')(x)
x = Dense(256, activation='relu', name='head_dense')(x)
x = Dropout(0.4, name='head_dropout')(x)
outputs = Dense(num_classes, activation='softmax', name='pred')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights("model_resnet101_penyakit_cabai.h5")
print("✅ Model berhasil dimuat.")

labels = ["Bercak", "Gemini", "Kriting", "Sehat"]
pages  = ["bercak.html", "gemini.html", "kriting.html", "sehat.html"]

# ================= PREDIKSI =================
def predict_disease(img_path):
    img = load_img(img_path, target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    pred = np.argmax(result, axis=1)[0]
    return labels[pred], pages[pred]

def predict_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (224,224))
    img = preprocess_input(resized)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    pred = np.argmax(result, axis=1)[0]
    confidence = np.max(result)
    return labels[pred], confidence

# ================= ROUTES =================
@app.route("/", methods=['GET'])
def main():
    cur = mysql.connection.cursor()
    cur.execute("SELECT nama, tanggal FROM riwayat ORDER BY tanggal ASC")
    rows = cur.fetchall()
    data = []
    for row in rows:
        data.append({
            'nama': row[0],
            'tanggal': str(row[1]),
        })
    cur.close()
    return render_template('dashboard.html', active_menu='dashboard', data=data)

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            flash("Masukkan gambar dulu!", "warning")
            return redirect(url_for('predict'))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        pred, output_page = predict_disease(file_path)
        # pastikan active_menu tetap 'predict'
        return render_template(output_page, pred_output=pred, user_image=file_path, active_menu='predict')
    return render_template('clasification.html', active_menu='predict')

@app.route("/save", methods=['POST'])
def save_history():
    name = request.form['name']
    tanggal = request.form['tanggal']
    waktu = request.form['waktu']
    gambar = request.form['gambar']  # base64 string atau path

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO riwayat (nama, tanggal, waktu, gambar)
        VALUES (%s, %s, %s, %s)
    """, (name, tanggal, waktu, gambar))
    mysql.connection.commit()
    cur.close()
    flash("Riwayat berhasil disimpan!", "success")
    return redirect(url_for('history'))

@app.route("/history", methods=['GET'])
def history():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM riwayat ORDER BY id DESC")
    rows = cur.fetchall()
    history_data = []
    for row in rows:
        history_data.append({
            'nama': row[1],
            'tanggal': row[2],
            'waktu': row[3],
            'gambar': row[4]
        })
    cur.close()
    return render_template('history.html', active_menu='history', history_data=history_data)

# ================= CAMERA STREAM =================
camera_active = False
def gen_frames():
    global camera_active
    cap = cv2.VideoCapture(0)
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        pred, conf = predict_frame(frame)
        label = f"{pred} - {conf:.2f}"
        if pred in ["Chili - bercak", "Chili - keriting"]:
            h, w, _ = frame.shape
            frame = cv2.rectangle(frame, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0,255,0), 2)
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

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

# ================= RUN FLASK =================
if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)
