from flask import Flask, request
import os
import json

app = Flask(__name__)
app.debug = True
app.config["UPLOAD_PATH"] = "/Users/sameal/Documents/PROJECT/zombie_enterprise/web/uploads"

@app.route("/")
def hello_world():
    return "Hello World!"

@app.route("/upload", methods=["GET", "POST"])
def upload():
    f = request.files["file"]
    key = request.form.get("suffixKey")
    upload_dir = os.path.join(app.config["UPLOAD_PATH"], key)
    if not os.path.exists(upload_dir):
        os.mkdir(upload_dir)
    upload_path = os.path.join(upload_dir, f.filename)
    f.save(upload_path)
    return "Success"

@app.route("/remove", methods=["GET", "POST"])
def remove():
    file_name = request.form.get("fileName")
    key = request.form.get("suffixKey")
    file_path = os.path.join(app.config["UPLOAD_PATH"], key, file_name)
    os.remove(file_path)
    return "Success"

@app.route("/predict", methods=["GET"])
def predict():
    return "Success"

if __name__ == "__main__":
    app.run(debug=True)
