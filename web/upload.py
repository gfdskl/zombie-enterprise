from flask import Flask, request
import os

app = Flask(__name__)
app.debug = True
app.secret_key = "helloworld"

@app.route("/")
def hello_world():
    return "Hello World!"

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]
        base_path = os.path.abspath(os.path.dirname(__file__))
        upload_path = os.path.join(base_path, "uploads", f.filename)
        f.save(upload_path)
        return "Success"

if __name__ == "__main__":
    app.run(debug=True)
