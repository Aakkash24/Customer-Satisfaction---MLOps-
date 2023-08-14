import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstact class for all models
    """
    @abstractmethod
    def train(self,X_train,Y_train):
        """
        Trains the model
        Args:   
            X_train  = Training data
            Y_train = Training labels
        """
        pass

class LinearRegressionModel(Model):
    def train(self,X_train,Y_train,**kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train,Y_train)
            logging.info("Model trained")
            return reg
        except Exception as e:
            logging.error(f"Error in training model : {e}")
            raise e
    