from flask import Flask, redirect, url_for, render_template, current_app, request, jsonify
from flask_cors import CORS, cross_origin
import os
import json
from werkzeug.utils import secure_filename
from frameDivider import devideFrames
from stance import classifyMainFrames
from TechniqueAnalyzer import analyseMainFrames
import DBService as DBservice

app = Flask (__name__)
app.debug =True

app.secret_key = "secret key"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['mp4', 'mov'])
cors = CORS(app, resources={r'/*': {"origins": '*'}})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def landing():
    print("welcome")
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("index.html")

@app.route('/frame', methods=['POST'])
@cross_origin(headers=['Content-Type'])
def frame():
    print("Dividing the frames", request.data.decode('UTF-8'))
    devideFrames(request.data.decode('UTF-8'))
    # process = Popen(['python', 'frame-output-stage1.py', '-v', request.data.decode('UTF-8')], stdout=PIPE, stderr=PIPE)
    # stdout, stderr = process.communicate()
    # print(stdout)
    # os.startfile('frame-output-stage1.py')
    # exec(open('frame-output-stage1.py').read())
    print("Dividing Done")
    print("Frame Available ? ",  os.path.exists(os.path.join(current_app.root_path, 'static/stage1-output-frames/frame1.jpg')))
    print("================================================================")
    # name = request.args.get("name", "World")
    # return f'Hello, {escape(name)}!'
    data = {'data': {
        'message': 'Divided'},
        'status': 200}
    return jsonify(data)

@app.route('/classifyFrames', methods=['GET'])
@cross_origin()
def classifyFrames():
    print("Detecting key-points on frames...")
    data = classifyMainFrames()
    # process = Popen(['python', 'deployStanceLMExecutionClassiyfier.py'], stdout=PIPE, stderr=PIPE)
    # stdout, stderr = process.communicate()
    # print(stdout.decode('UTF-8'))
    print("Classifying Done...")
    print("================================================================")

    data = {'data': data,
            'status': 200}
    return jsonify(data)


@app.route('/getAnalysis', methods=['POST'])
@cross_origin()
def getAnalysis():
    data = json.loads(request.data.decode('UTF-8'))
    print("Frames to be analysed", data)
    analyzedData = analyseMainFrames(data)
    data = {'data': analyzedData,
            'status': 200}
    return jsonify(data)

@app.route('/loginUser', methods=['POST','GET'])
@cross_origin()
def loginUser():
    if request.method == "POST":
        data = json.loads(request.data.decode('UTF-8'))
        print("Login this: ", data)
        result = DBservice.login_user(data["email"],data["password"])
        data = {'data': result,
                'status': 200}
        return jsonify(data)
    else:
        return render_template(
                "login.html"
            )

@app.route('/saveStroke', methods=['POST'])
@cross_origin()
def saveStroke():
    data = json.loads(request.data.decode('UTF-8'))
    print("save this: ", data)
    result = DBservice.save_stroke(data["id"],data["stance"],data["execution"])
    data = {'data': result,
            'status': 200}
    return jsonify(data)

@app.route('/getAnalyticsPercentage', methods=['POST'])
@cross_origin()
def getAnalyticsPercentage():
    data = request.data.decode('UTF-8')
    print("Login this: ", data)
    result = DBservice.getAnalyticsData(data)
    data = {'data': result,
            'status': 200}
    return jsonify(data)

@app.route('/viewAnalitics',  methods=['GET'])
def getviewAnalitics():
    print("analysis")
    return render_template("analytics.html")
            

@app.route('/videoUpload', methods=['POST'])
@cross_origin(headers=['Content-Type'])
def videoUpload():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files[]')
    filename = ''
    errors = {}
    success = False

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("filename", filename)
            file.save(os.path.join(current_app.root_path,'static/upload_folder',filename))
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        errors['data'] = filename
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message': 'Files successfully uploaded', 'data': filename})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run()

