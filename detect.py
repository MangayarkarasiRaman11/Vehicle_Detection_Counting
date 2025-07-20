import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os
from time import sleep
from flask import Flask, render_template, request, redirect, url_for, session
from functools import wraps

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

largura_min = 80
altura_min = 80
offset = 6
pos_linha = 550
delay = 60


def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

users = {}

# Authentication Decorator to Restrict Access to Prediction Page
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))  # Redirect to login page if not logged in
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/help')
def help():
    return render_template('help.html')



@app.route('/uploader', methods=['POST'])
def uploader_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return redirect(url_for('process_video', filename=file.filename))


@app.route('/process/<filename>')
def process_video(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(file_path)
    subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()
    carros = 0
    detec = []

    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret:
            break
        sleep(1 / delay)
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = subtracao.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        contorno, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (176, 130, 39), 2)
        for c in contorno:
            (x, y, w, h) = cv2.boundingRect(c)
            if w >= largura_min and h >= altura_min:
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                centro = pega_centro(x, y, w, h)
                detec.append(centro)
                cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

                for (cx, cy) in detec:
                    if (cy < (pos_linha + offset)) and (cy > (pos_linha - offset)):
                        carros += 1
                        cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
                        detec.remove((cx, cy))
                        print("No. of cars detected :", carros)

        cv2.putText(frame1, f"VEHICLE COUNT: {carros}", (320, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Vehicle Detection", frame1)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return f"Processing Completed. Vehicles detected: {carros}"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email] == password:
            session['user'] = email  # Store user session
            return redirect(url_for('upload_file'))  # Redirect to prediction page after login
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if email in users:
            return render_template("register.html", error="Email already registered")
        users[email] = password
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user session
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
