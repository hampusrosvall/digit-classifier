from flask import Flask 
import json

app = Flask(__name__)

@app.route('/_health')
def index():
    return json.dumps({'status' : {'Feeling good' : 200}}, indent = 4)

@app.route('/predict')
def predict(): 
    pass
    
if __name__ == '__main__': 
    app.run(debug = True, host = 'localhost', port = 5000)