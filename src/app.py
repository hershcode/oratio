from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from models.models import Oratio

app = Flask(__name__)

UPLOAD_FOLDER= "data/upload"
app.config['SECRET_KEY'] = 'supersecretkey'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=['GET', "POST"])
def index():
    if request.method == "POST":
        print("Form Data received")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        if file.filename in os.listdir(UPLOAD_FOLDER):
            
            model = Oratio()
            word = model.predict('data/upload/'+file.filename)

            return word

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True) 