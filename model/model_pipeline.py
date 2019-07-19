import keras
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import Adam 

class ModelPipeline: 
    def __init__(self): 
        self.X_train = None 
        self.X_test = None
        self.y_train = None
        self.y_test = None 

    def read_data(self): 
        """ 

        """ 
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def build_model(self): 
        """


        """
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

        # layers with 32 filters
        model.add(Conv2D(filters = 32, kernel_size = filter_shape, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = pooling_shape))

        # layers with 64 filters
        model.add(Conv2D(filters = 64, kernel_size = filter_shape, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = pooling_shape))

        model.add(Flatten())
        model.add(Dense(1, activation = 'softmax'))

        return model

        
if __name__ == '__main__': 
    mp = ModelPipeline()
    mp.build_model()