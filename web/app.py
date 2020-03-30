from flask import Flask, request, send_file
import os
import sys
from multiprocessing import Process
import json
from waiting import wait, TimeoutExpired

LIB_ROOT = "/Users/sameal/Documents/PROJECT/zombie_enterprise/code"
sys.path.append(LIB_ROOT)
import predict_tools

app = Flask(__name__)
app.debug = True
app.config["UPLOAD_PATH"] = "/Users/sameal/Documents/PROJECT/zombie_enterprise/web/uploads"


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["file"]
    key = request.form.get("suffixKey")
    upload_dir = os.path.join(app.config["UPLOAD_PATH"], key)
    if not os.path.exists(upload_dir):
        os.mkdir(upload_dir)
    upload_path = os.path.join(upload_dir, f.filename)
    f.save(upload_path)
    return "Success"

@app.route("/remove", methods=["POST"])
def remove():
    file_name = request.form.get("fileName")
    key = request.form.get("suffixKey")
    file_path = os.path.join(app.config["UPLOAD_PATH"], key, file_name)
    os.remove(file_path)
    pred_path = os.path.join(app.config["UPLOAD_PATH"], key, "result.csv")
    if os.path.exists(pred_path):
        os.remove(pred_path)
    return "Success"

@app.route("/predict", methods=["POST"])
def predict():
    key = request.form.get("suffixKey")
    upload_dir = os.path.join(app.config["UPLOAD_PATH"], key)
    res_path = os.path.join(upload_dir, "result.csv")
    if not os.path.exists(res_path):
        p = Process(target=predict_tools.analyse, args=(upload_dir,))
        p.start()
        predict_tools.predict(upload_dir)
    try:
        return send_file(res_path, mimetype="text/csv", as_attachment=True, attachment_filename="result.csv")
    except Exception as e:
        app.log_exception(e)

@app.route("/search", methods=["GET"])
def search():
    key = request.args.get("suffixKey")
    search_id = request.args.get("id")
    upload_dir = os.path.join(app.config["UPLOAD_PATH"], key)
    portrait_path = os.path.join(upload_dir, "portrait.csv")
    try:
        wait(lambda: os.path.exists(portrait_path), timeout_seconds=10)
        label = predict_tools.search(upload_dir, search_id)
        if label is None:
            return ""
        return label
    except TimeoutExpired:
        return "Timeout", 408

@app.route("/chart", methods=["GET"])
def chart():
    key = request.args.get("suffixKey")
    search_id = request.args.get("id")
    byclass = request.args.get("class")
    upload_dir = os.path.join(app.config["UPLOAD_PATH"], key)
    data = predict_tools.chart(upload_dir, search_id, byclass)
    if data is None:
        return ""
    return data

if __name__ == "__main__":
    app.run(debug=True)
