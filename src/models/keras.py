""" Link travel time models build with keras and tensorflow. """

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from kerastuner.tuners import RandomSearch, Hyperband

class KerasBaseModel():
    
    def __init__(self):
        self.dense_n = 0
        self.epochs = 100
        self.batch_size = 1000      
        self.y_names = ['link_travel_time']
    
    def weighted_mae_loss(self, y_true, y_pred, weights):
        return K.mean(K.abs(y_true - y_pred) * weights)

    def weighted_mse_loss(self, y_true, y_pred, weights):
        return K.mean(K.pow(y_true - y_pred, 2) * weights)
    
    def dense(self, n, activation = 'relu'):
        self.dense_n += 1
        return keras.layers.Dense(
            units=n, 
            name=f'dense_{self.dense_n}', 
            activation = activation, 
            use_bias=True)
    
    def extract_y(self, ts):
        return ts[self.y_names]
    
    @abstractmethod
    def build_model(self, hp):
        pass

    def transform_y(self, ts, is_training):
        return ts[self.y_names].values
    
    def inverse_transform_y(self, y_pred, ix):
        return pd.DataFrame(data = y_pred, index = ix, columns=self.y_names)
    
    @abstractmethod
    def transform(self, x, is_training):
        pass
    
    def predict(self, input_data):
        return self.inverse_transform_y(self.model.predict(self.transform(input_data, is_training = False)), input_data.index) 
        
    def build_and_train(self, train_data, val_data, hp = None, callbacks = None, verbose = 0):
        x_train = self.transform(train_data.drop(self.y_names, axis = 1), is_training = True)
        y_train = np.squeeze(self.transform_y(train_data, is_training = True))
        
        if val_data is not None:
            x_val =  self.transform(val_data.drop(self.y_names, axis = 1), is_training = False)
            y_val = np.squeeze(self.transform_y(val_data, is_training = False))
            validation_data = x_val, y_val
        else:
            validation_data = None
        
        if 'batch_size' in hp:
            batch_size = hp['batch_size']
        else:
            batch_size = self.batch_size

        if 'epochs' in hp:
            epochs = hp['epochs']
        else:
            epochs = self.epochs
            
        self.model = self.build_model(hp)        
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_data=validation_data, verbose = verbose)
    
class BaselineModel(KerasBaseModel):
    
    def __init__(self):
        super(BaselineModel, self).__init__()
    
    def choose_tod_bins(self, ts):
        bins = []
        delta_t = pd.DataFrame(index = ts.index[1:], data = np.diff(ts.index) / pd.Timedelta('1H'))
        mean_hour_interval_count = np.ceil(1 / delta_t.groupby(delta_t.index.hour)[0].mean())
        for hour, hour_interval_count in mean_hour_interval_count.iteritems():
            # We really dont want to deal floating point intervals - pick some intervals that are representable with second granuality
            if hour_interval_count > 10:
                hour_interval_count = 12
            elif 7 <= hour_interval_count and hour_interval_count <= 9:
                hour_interval_count = 8
            elif hour_interval_count == 5:
                hour_interval_count = 6
            elif hour_interval_count == 3:
                hour_interval_count = 4
            elif hour_interval_count < 1:
                hour_interval_count = 1
            bins.extend(list((60*60*hour + np.arange(hour_interval_count) * (60*60 / hour_interval_count)).astype(int)))
        self.tod_bins = np.array(bins)
        self.tod_bins_n = len(bins)
        
        self.mean = ts[self.y_names].mean().values
        self.std = ts[self.y_names].std().values
    
    def transform_y(self, ts, is_training):
        y = super().transform_y(ts, is_training)
        y[:, 0] = (y[:, 0] - self.mean[0]) / self.std[0]
        return y
    
    def transform(self, x, is_training):       
        tod = (60 * 60 * x.index.hour + 60 * x.index.minute + x.index.second).values            
        X_dow_train = np.eye(7)[x.index.dayofweek]
        X_tod_train = np.eye(self.tod_bins_n)[np.digitize(tod, self.tod_bins) - 1]
        return [X_dow_train, X_tod_train]
    
    """
    def transform(self, ts, is_training):
        freq = pd.to_timedelta('1min')
        timesteps_per_day = pd.to_timedelta('24H') / freq
        x = ts.index.dayofweek.values * timesteps_per_day + ts.index.map(lambda x: pd.to_timedelta(x.time().isoformat())).values / freq
        x_2 = ts.index.map(lambda x: pd.to_timedelta(x.time().isoformat())).values / freq
        return np.stack([
            np.sin(x / (7*24*60) * 2 * np.pi),
            np.cos(x / (7*24*60) * 2 * np.pi),
            np.sin(x_2 / (24*60) * 2 * np.pi),
            np.cos(x_2 / (24*60) * 2 * np.pi)
        ], axis = 1)
    """
    
    def inverse_transform_y(self, y_pred, ix):
        y_pred = y_pred * self.std + self.mean
        return super().inverse_transform_y(y_pred, ix)
    
    def build_model(self, hp):
        input_dow = keras.layers.Input(shape = (7,), name = 'input_dow')
        input_tod = keras.layers.Input(shape = (self.tod_bins_n,), name = 'input_tod')
        #input_tod_dow = keras.layers.Input(shape = (4,), name = 'input_tod_dow')
        
        do_prop = hp.Choice('dropout', [0., .1, .2])
        
        x = keras.layers.Concatenate(name = 'concat_dow_tod', axis = 1)([input_dow, input_tod])
        #x = input_tod_dow
                
        for i in range(hp.Int('num_layers', 1, 3, default = 1)):
            if i > 0 and hp.Choice('batch_norm', [True, False]):
                x = keras.layers.BatchNormalization(name = f'bn_{i + 1}')(x)
            x = keras.layers.Dense(units=hp.Int(f'fc_{i + 1}_units',
                                                min_value=100,
                                                max_value=500,
                                                step=50,
                                                default = 250),
                                   activation=hp.Choice(f'fc_{i + 1}_activation', ['relu', 'tanh']), name = f'fc_1_{i + 1}')(x)
            x = keras.layers.Dropout(do_prop, name = f'dropout_{i + 1}')(x)
        x = keras.layers.Dense(1, activation = 'linear', name = 'fc_out')(x)
        output = x
        
        model = keras.Model(inputs=[input_dow, input_tod], outputs=[output])
        model.compile(
            optimizer=tf.train.RMSPropOptimizer(
                learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3]),
                momentum = hp.Choice('momentum', [0.65, 0.75, 0.85]),
                decay = hp.Choice('decay', [0.95, 1.])),
            loss='mse')
        
        return model
    
class DayTypeEmbeddingsModel(KerasBaseModel):
    
    def __init__(self, daytype_components, cal):
        super(DayTypeEmbeddingsModel, self).__init__()
        self.daytype_components = daytype_components
        self.cal = cal
            
    def choose_tod_bins(self, ts):
        #"""
        bins = []
        delta_t = pd.DataFrame(index = ts.index[1:], data = np.diff(ts.index) / pd.Timedelta('1H'))
        mean_hour_interval_count = np.ceil(1 / delta_t.groupby(delta_t.index.hour)[0].mean())
        for hour, hour_interval_count in mean_hour_interval_count.iteritems():
            # We really dont want to deal floating point intervals - pick some intervals that are representable with second granuality
            if hour_interval_count > 10:
                hour_interval_count = 12
            elif 7 <= hour_interval_count and hour_interval_count <= 9:
                hour_interval_count = 8
            elif hour_interval_count == 5:
                hour_interval_count = 6
            elif hour_interval_count == 3:
                hour_interval_count = 4
            elif hour_interval_count < 1:
                hour_interval_count = 1
            bins.extend(list((60*60*hour + np.arange(hour_interval_count) * (60*60 / hour_interval_count)).astype(int)))
        self.tod_bins = np.array(bins)
        self.tod_bins_n = len(bins)
        #"""
        self.mean = ts[self.y_names].mean().values
        self.std = ts[self.y_names].std().values
            
    def transform_y(self, ts, is_training):
        y = super().transform_y(ts, is_training)
        if 'mean' in ts.columns and 'std' in ts.columns:
            y = (y[:, 0] - ts['mean'].fillna(self.mean[0]).values) / self.std[0]
            ts['mean'] = ts['mean'].fillna(self.mean[0])
            ts['std'] = self.std[0]
        else:
            print(y.shape)
            y = (y[:, 0] - self.mean[0]) / self.std[0]
            ts['mean'] = self.mean[0]
            ts['std'] = self.std[0]
        return y
    
    def inverse_transform_y(self, y_pred, ix):
        y_pred = y_pred * self.std + self.mean
        return super().inverse_transform_y(y_pred, ix)
            
    def transform(self, ts, is_training):
        X_daytype_embedding = self.daytype_components.loc[self.cal.loc[ts.index.date, 'day_type']].values
        
        # Calculate "seconds since midnight"
        tod = (60 * 60 * ts.index.hour + 60 * ts.index.minute + ts.index.second).values
        X_dow = np.eye(7)[ts.index.dayofweek]
        X_tod = np.eye(self.tod_bins_n)[np.digitize(tod, self.tod_bins) - 1]
        """
        freq = pd.to_timedelta('1min')
        timesteps_per_day = pd.to_timedelta('24H') / freq
        x = ts.index.dayofweek.values * timesteps_per_day + ts.index.map(lambda x: pd.to_timedelta(x.time().isoformat())).values / freq
        x_2 = ts.index.map(lambda x: pd.to_timedelta(x.time().isoformat())).values / freq
        X_tod_dow = np.stack([
            np.sin(x / (7*24*60) * 2 * np.pi),
            np.cos(x / (7*24*60) * 2 * np.pi),
            np.sin(x_2 / (24*60) * 2 * np.pi),
            np.cos(x_2 / (24*60) * 2 * np.pi)
        ], axis = 1)
        """
        
        return [X_daytype_embedding, X_dow, X_tod]
    
    def build_model(self, hp):
        input_embedding = keras.layers.Input(shape = (4, ), name = 'input_embedding')
        input_dow = keras.layers.Input(shape = (7,), name = 'input_dow')
        input_tod = keras.layers.Input(shape = (self.tod_bins_n,), name = 'input_tod')
        #input_tod_dow = keras.layers.Input(shape = (4,), name = 'input_tod_dow')
        
        do_prop = hp.Choice('dropout', [0., .1, .2])
        #x = keras.layers.Concatenate(name = 'concat_dow_tod', axis = 1)([input_dow, input_tod])
        x = keras.layers.Dense(units=hp.Int(f'fc_1_units',
                                            min_value=100,
                                            max_value=500,
                                            step=50,
                                            default = 250),
                               activation=hp.Choice('fc_1_activation', ['relu', 'tanh']), name = f'fc_1')(input_embedding)
        x = keras.layers.Dropout(do_prop, name = 'dropout_1')(x)
        x = keras.layers.Concatenate(name = 'concat_embedding', axis = 1)([x, input_dow, input_tod])
        for i in range(hp.Int('num_layers', 1, 2, default = 1)):
            if hp.Choice('batch_norm', [True, False]):
                x = keras.layers.BatchNormalization(name = f'bn_{i + 2}')(x)
            x = keras.layers.Dense(units=hp.Int(f'fc_{i + 2}_units',
                                                min_value=100,
                                                max_value=500,
                                                step=50,
                                                default = 250),
                                   activation=hp.Choice(f'fc_{i + 2}_activation', ['relu', 'tanh']), name = f'fc_1_{i + 2}')(x)
            x = keras.layers.Dropout(do_prop, name = f'dropout_{i + 2}')(x)
        x = keras.layers.Dense(1, activation = 'linear', name = 'fc_out')(x)
        output_1 = x
            
        model = keras.Model(inputs=[input_embedding, input_dow, input_tod], outputs=[output_1])
        model.compile(
            optimizer=tf.train.RMSPropOptimizer(
                learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3]),
                momentum = hp.Choice('momentum', [0.65, 0.75, 0.85]),
                decay = hp.Choice('decay', [0.95, 1.])),
            loss='mse')
        
        return model   

class HyperbandBatchSizeTuner(Hyperband):
    """ This is a small extension to the Hyperband tuner witch allow tuning batch size also. """
    
    def __init__(self,
                 *args, **kwargs):
        self.batch_sizes = kwargs.pop('batch_sizes')
        super(HyperbandBatchSizeTuner, self).__init__(*args, **kwargs)
    
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', self.batch_sizes)
        super(HyperbandBatchSizeTuner, self).run_trial(trial, *args, **kwargs)
