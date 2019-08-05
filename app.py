from flask import Flask, request
from flask_restplus import Resource, Api
from PIL import Image

app = Flask(__name__)
api = Api(app)

@api.route('/_health')
class Health(Resource):
    def get(self):
        return {'Feeling': 'Good'}

@api.route('/predict')
class Predict(Resource):
    def post(self):
        print(request.files['file'])
        #img = Image.open(request.files['file'])
        return 'Success!'

if __name__ == '__main__':
    app.run(debug=True)