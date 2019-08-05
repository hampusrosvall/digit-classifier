from flask import Flask, request
from flask_restplus import Resource, Api
from PIL import Image
import keras
from keras import backend as K
import numpy as np
from keras.optimizers import RMSprop

app = Flask(__name__)
api = Api(app)

@api.route('/_health')
class Health(Resource):
    def get(self):
        return {'Feeling': 'Good'}

@api.route('/predict')
class Predict(Resource):
    def post(self):
        model = keras.models.load_model('./models/my_model.h5')

        # define the optimizer
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # compile the model
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


        img = Image.open(request.files['file'])
        img = img.resize((28, 28))
        img = np.array(img)[:, :, 0].reshape(-1, 28, 28, 1) / 255

        K.clear_session()
        prediction = model.predict(img)
        K.clear_session()

        prediction_number = np.argmax(prediction)
        return 'You entered a: {}'.format(prediction_number)

if __name__ == '__main__':
    app.run(debug=True)