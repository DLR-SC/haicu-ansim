
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.dummy import  DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot

class Baseline():
    def __init__(self,  X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def dummy_train_test(self, strategy='mean'):
        clf = DummyRegressor(strategy=strategy)
        '''
            “mean”: always predicts the mean of the training set
        
            “median”: always predicts the median of the training set
        
            “quantile”: always predicts a specified quantile of the training set, provided with the quantile parameter.
        
            “constant”: always predicts a constant value that is provided by the user.
        '''

        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        result_dic = {}

        result_dic['mse'] = round(mean_squared_error(y_pred=y_pred, y_true=self.y_test), 4)
        result_dic['mae'] = round(mean_absolute_error(y_pred=y_pred, y_true=self.y_test),  4)
        return result_dic


class GenericModel():
    def __init__(self, training_dataset, test_dataset, input_shape_instances, input_shape_features,
                 output_shape=2,
                 lr=1e-3, epochs=100, loss="mse", metrics=["mse", "mae"]):
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.input_shape_instances = input_shape_instances  # should be equal to the window size
        self.input_shape_features = input_shape_features  # should be equal to the # of x columns
        self.loss = loss
        self.metrics = metrics
        self.lr = lr
        self.epochs = epochs
        self.output_shape = output_shape
        self.model = None

    def _reset(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(2209)
        np.random.seed(2209)

        tf.keras.backend.clear_session()

    def init_model(self):
        print('ERROR - init model should be called from a child class')
        pass

    def evaluate(self, verbose=1):
        self.model.evaluate(self.test_dataset, verbose=verbose)

    def evaluate_saved_model(self, model_path, verbose=0):
        # load the saved model
        saved_model = tf.keras.models.load_model(model_path)
        # evaluate the model
        train_eval = saved_model.evaluate(self.training_dataset, verbose=verbose)
        test_eval = saved_model.evaluate(self.test_dataset, verbose=verbose)
        print('Train: loss, mse mae --> ', str(train_eval))
        print('Test: loss, mse mae --> ', str(test_eval))

    def train(self, stop_early = False, best_model_name = None, plot_loss= False,
              patience=5, min_delta=0.05, verbose=1):
        self._reset()
        self.init_model()
        callbacks = []

        if stop_early:
            if best_model_name is None or best_model_name == '':
                print('please set the best model name - in case of early stopping, the best model will be saved')
            early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, min_delta=min_delta)
            model_checkpoint = ModelCheckpoint(best_model_name+'.h5', monitor='val_loss', mode='min',
                                               verbose=1, patience=patience, min_delta=min_delta,
                                               save_best_only=True)
            callbacks = [early_stopping, model_checkpoint]

        self.model.compile(loss= self.loss, optimizer=tf.keras.optimizers.Adam(lr=self.lr), metrics=self.metrics
                           )
        history = self.model.fit(self.training_dataset, validation_data=self.test_dataset, epochs=self.epochs,
                                 callbacks=callbacks,
                                 verbose=verbose)

        if plot_loss:
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()
        return history


class SequentialModel(GenericModel):

    def init_model(self):
        print('Initializing the 3 layers sequential model')
        self.model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(self.input_shape_instances,
                                                         self.input_shape_features)),#reshapes to 15000
                    tf.keras.layers.Dense(100,
                                          activation="relu"),
                    tf.keras.layers.Dense(10, activation="relu"),
                    tf.keras.layers.Dense(self.output_shape)
    ])


class Lstm(GenericModel):

    def init_model(self):
        print('Initializing the Lstm model')

        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
            #                    input_shape=[None, 5]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True),
                                          input_shape=[self.input_shape_instances, self.input_shape_features]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(self.output_shape),
            # tf.keras.layers.Lambda(lambda x: x * 10.0)
        ])


    def tune_lr(self):
        self._reset()
        self.init_model()

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: self.lr * 10 ** (epoch / 20))
        optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.model.compile(loss=tf.keras.losses.Huber(),  # less sensitive to outliers
                      optimizer=optimizer,
                      metrics=self.metrics)
        history = self.model.fit(self.training_dataset, epochs=self.epochs, callbacks=[lr_schedule])

        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([self.lr, 1e-1, 0, max(history.history["loss"]) + 10])
        plt.xlabel('learning rate')
        plt.ylabel('loss (Huber)')
        return history

    def get_best_epoch(self, history):
        best_epoch = np.argmin(np.array(history.history['loss']))
        return {'best_epoch': best_epoch, 'loss': history.history['loss'][best_epoch]}

    def plot_history(history, zoom=False):
        # -----------------------------------------------------------
        # Retrieve a list of list results on training and test data
        # sets for each training epoch
        # -----------------------------------------------------------
        mae = history.history['mae']
        loss = history.history['loss']

        epochs = range(len(loss))  # Get number of epochs

        # ------------------------------------------------
        # Plot MAE and Loss
        # ------------------------------------------------
        plt.plot(epochs, mae, 'r')
        # plt.plot(epochs, loss, 'b')
        plt.title('MAE')
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        # plt.legend(["MAE", "Loss"])

        plt.figure()
        plt.plot(epochs, loss, 'b')
        plt.title('Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.figure()
        if zoom:
            epochs_zoom = epochs[200:]
            mae_zoom = mae[200:]
            loss_zoom = loss[200:]

            # ------------------------------------------------
            # Plot Zoomed MAE and Loss
            # ------------------------------------------------
            plt.plot(epochs_zoom, mae_zoom, 'r')
            plt.plot(epochs_zoom, loss_zoom, 'b')
            plt.title('MAE and Loss')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(["MAE", "Loss"])

            plt.figure()