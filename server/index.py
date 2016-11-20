from flask import Flask, json, redirect, url_for, request, send_from_directory, render_template
import csv, os
from flask.ext.cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import classification

PATH_TO_UPLOADS = "./uploads"
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__, static_url_path='/statics/')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = PATH_TO_UPLOADS
classificationResult = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/<path:path>")
def root(path):
    return send_from_directory('statics', path)

@app.route("/result",  methods = ['POST', 'GET'])
def showResult():
    return json.jsonify(results=classificationResult)

@app.route('/upload', methods = ['POST', 'GET'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv'))
            global classificationResult
            classificationResult = classification.classify()
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
