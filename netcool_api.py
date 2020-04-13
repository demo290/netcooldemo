import os
from flask import Flask, redirect , url_for, render_template , request, jsonify
from werkzeug.utils import secure_filename
from model_pred import doc_path
from flasgger import Swagger

UPLOAD_FOLDER = "demo290/netcooldemo/upload"

app = Flask(__name__)
swagger = Swagger(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('flask.html')

@app.route('/upload', methods=['POST'])
def upload():
    doc = request.files['myFile']
    filename = secure_filename(doc.filename)
    file_path = doc.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # calling function for prediction
    result = doc_path(UPLOAD_FOLDER+'/'+filename)
    # print(result)
    return result

    
if __name__ == '__main__':
    app.run(debug=True)
