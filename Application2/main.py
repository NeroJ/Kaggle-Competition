from __future__ import print_function
from flask import Flask
from flask import render_template
from flask import request
from flask import abort, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
from fraudModel.fraud import *
import sys
import copy
import time

UPLOAD_FOLDER = os.getcwd()+'/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        modelType = request.form['select']
        #print(modelType,file=sys.stderr)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #print(filename,file=sys.stderr)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            Use_Model(Clustering=True, baseDir = os.getcwd(), modelType = modelType, filename = filename)
            if modelType == 'RF':
                time.sleep(6.5)
            elif modelType == 'LRsgd':
                time.sleep(4.5)
            elif modelType == 'LRlbfgs':
                time.sleep(5)
            elif modelType == 'GBDT':
                time.sleep(5.8)
            elif modelType == 'SMV':
                time.sleep(5.0)
            else:
                pass    
            return redirect(url_for('uploaded_file',
                                    filename=filename+'_'+modelType +'_predicted.csv'))
    return render_template('hello.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
  app.run()
