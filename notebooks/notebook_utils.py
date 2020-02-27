import numpy as np
import pandas as pd
from scipy import interpolate

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from IPython.display import clear_output
   
tf.logging.set_verbosity(tf.logging.ERROR)

def td_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('D',      60*60*24),
        ('H',      60*60),
        ('min',    60)
    ]

    strings=[]
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value , seconds = divmod(seconds, period_seconds)
            strings.append("%s%s" % (period_value, period_name))

    return "+".join(strings)

# Visualization functions
def highlight_min(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.min()
    return ['color: red; font-weight: bold' if v else 'color: #666' for v in is_min]

def run_rolling_window_tss(
    tss,                        # Time Series
    fun,                        # Function to run    
    test_start,                 # Test start
    train_start = None,         # Train start (optional)
    test_window_roll = None,    # How much is the test window rolled at each run
    test_window_horizon = None, # How large is the test window
    test_window_boostrap_time = None,
    test_window_predict_horizon = None,
    slide_train = False,        # Shuld the train window allso roll with same roll as test
    n_runs = 1,                 # Number of runs    
    **kwargs):
    """Runs several iterations of the test function on the data frame to simulate real world application."""    

    results = []
    data_freq = tss[0].index[1] - tss[0].index[0]
    test_window_boostrap_time_steps = int(test_window_boostrap_time / data_freq)
    predict_time_steps = int(test_window_predict_horizon / data_freq)
    
    for i in range(n_runs):
        run_train_start = tss[0].index.min() if train_start is None else train_start
        run_test_start = test_start + i * test_window_roll        
        
        if slide_train:
            run_train_start = run_train_start + i * test_window_roll
            
        if test_window_boostrap_time is None:
            test_window_boostrap_time = pd.to_timedelta(0)
        
        if test_window_predict_horizon is None:
            test_window_predict_horizon = pd.to_timedelta(0)
        
        delta_min = pd.to_timedelta('1ns')
        
        # split in train and test
        train = [ts[run_train_start:run_test_start - delta_min].copy() for ts in tss]
        test = [ts[run_test_start - test_window_boostrap_time:run_test_start + test_window_horizon + test_window_predict_horizon - data_freq - delta_min].copy() for ts in tss]

        run = {       
            'data_freq': data_freq,
            'train_size': train[0].shape[0],
            'test_size': test[0].shape[0],
            'train_start': run_train_start,
            'test_start': run_test_start,
            'train_key': '{0}-{1:%Y%m%dT%H%M}-{2}'.format(td_format(data_freq), run_train_start, td_format(run_test_start - run_train_start)),
            'test_key': '{0}x{1}-{2:%Y%m%dT%H%M}-{3}'.format(predict_time_steps, td_format(data_freq), run_test_start, td_format(test_window_horizon)),
            'test_window_boostrap_time': test_window_boostrap_time,
            'test_window_boostrap_time_steps': test_window_boostrap_time_steps,
            'predict_time_horizon': test_window_predict_horizon,
            'predict_time_steps': predict_time_steps,
            'slide_train': slide_train,
        }
        results.append(fun(run, train, test, **kwargs))
        
    return results

def transform_ix(ix, ref_time, freq):
    """Transform a DateTimeIndex to a numeric value."""
    return (ix - ref_time).values / freq

def smooth_align(in_ix, in_y, out_ix, freq, smooth = 50):
    ref_time = in_ix.min()
    in_x = transform_ix(in_ix, ref_time, freq)
    out_x = transform_ix(out_ix, ref_time, freq)
    rbf = interpolate.Rbf(in_x, in_y, smooth = smooth)
    return rbf(out_x)

def smooth_align_df(df, freq, smooth = 50):
    freq = pd.to_timedelta(freq)
    ix = pd.date_range(df.index.min().date(), df.index.max().date() + pd.DateOffset(1) - freq, freq=freq)
    return pd.DataFrame(data = smooth_align(in_ix=df.index, in_y=df.iloc[:, 0].values, out_ix=ix, freq=freq, smooth=smooth), index=ix, columns=df.columns[:1])

def init_tf_session(per_process_gpu_memory_fraction = 0.5, gpu_list = '0'):
    sess = tf.get_default_session()
    # close current session
    if sess is not None:
        sess.close()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    config.gpu_options.visible_device_list = gpu_list
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    return sess

class PlotLosses(keras.callbacks.Callback):
    
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []    
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        f, (ax1) = self.plt.subplots(figsize = (16, 9))
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        #ax1.set_title('loss: {.2f}, val_loss: {.2f}'.format(logs.get('loss')[0], logs.get('val_loss')))
        ax1.set_title('loss: {:.2f}'.format(logs.get('loss')))
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        self.plt.show();
