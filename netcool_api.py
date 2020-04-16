import os
from flask import Flask, redirect , url_for, render_template , request, jsonify, send_file
from werkzeug.utils import secure_filename
from model_pred import doc_path


UPLOAD_FOLDER = "upload"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('flask.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    doc = request.files['myFile']
    filename = secure_filename(doc.filename)
    file_path = doc.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # calling function for prediction
    result = doc_path(UPLOAD_FOLDER+'/'+filename)
    # print(result)
    return result
@app.route('/download', methods=['GET','POST'])
def download():
    return send_file('output.csv', as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=True)
