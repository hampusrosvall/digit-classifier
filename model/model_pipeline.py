import keras
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop

class ModelPipeline: 
    def __init__(self): 
        pass

    def read_data(self): 
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        return (X_train, y_train), (X_test, y_test)

    def build_model(self): 
        # initialize model 
        model = Sequential()

        # initialize parameters 
        pooling_shape = (2, 2)
        filter_shape = (3, 3)
        input_shape = (28, 28, 1) 

        # build CNN architecture 

        # layers with 16 filters 
        model.add(Conv2D(input_shape = input_shape, filters = 16, kernel_size = filter_shape, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = pooling_shape))
        model.add(Dropout(0.2))

        # layers with 32 filters
        model.add(Conv2D(filters = 64, kernel_size = filter_shape, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = pooling_shape))
        model.add(Dropout(0.2))

        # layers with 64 filters
        model.add(Conv2D(filters = 128, kernel_size = filter_shape, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = pooling_shape))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation = "softmax"))

        return model

    
    def model_training_pipeline(self): 
        # extract the data 
        (X_train, y_train), (X_test, y_test) = self.read_data()

        # one-hot encode the labels 
        (y_train, y_test) = (to_categorical(y_train, num_classes = 10),
                                     to_categorical(y_test, num_classes = 10))

        # reshape X_train, X_test to 4D tensors 
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        # build the model 
        model = self.build_model()

        # define the optimizer 
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # compile the model 
        model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

        # define imagedatagenerator 
        train_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
        )

        # fit the generator 
        train_datagen.fit(X_train)

        # fit the model 
        batch_size = 128
        steps_per_epoch = X_train.shape[0] // batch_size 
        epochs = 2

        model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch, epochs=epochs, verbose = 0,
                    validation_data = (X_test, y_test))

        model.save('../models/my_model.h5')
      
if __name__ == '__main__': 
    mp = ModelPipeline()
    mp.model_training_pipeline()