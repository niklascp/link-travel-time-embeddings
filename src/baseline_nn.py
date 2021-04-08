""" 
Baseline model using simple artificial intelligence neural network. 
Input is only day of week and time of day (in dynamic intervals)
"""

import os
import argparse
import dateutil as dt
import json

import numpy as np
import pandas as pd

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

from helpers import api, data_utils
from models.keras import BaselineModel, HyperbandBatchSizeTuner

parser = argparse.ArgumentParser(description='Trains and tests link travel time baseline model.')
parser.add_argument('group', help='unique link reference.')
parser.add_argument('time', help='datetime in isoformat.')
parser.add_argument('--tune', default=False, action='store_true')
parser.add_argument('--gpu', default='0', help='gpu (or list of gpus) to run on')
args = parser.parse_args()

config = tf.ConfigProto(
    intra_op_parallelism_threads = 0,
    inter_op_parallelism_threads = 0,
    allow_soft_placement = True
)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = args.gpu
sess = tf.Session(config=config)
K.set_session(sess)

with open(f'../output/links_{args.group}.txt') as f:
    link_refs = f.readlines()
    link_refs = [x.strip() for x in link_refs if not x.startswith('#')]

time = pd.to_datetime(args.time)

MODEL_NAME = 'baseline_nn'
N_NORMAL_DAYS = 28
N_PRED_DAYS = 7

HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
SEED = 42

test_end_time = time + pd.DateOffset(days = N_PRED_DAYS)

cal_train = api.calendar((time - pd.to_timedelta('1Y')).date(), to_date=time.date())
cal_test = api.calendar(from_date=time.date(), to_date=test_end_time.date())

for link_ref in link_refs:
    print('link_ref', link_ref)

    link_ref_slug = link_ref.replace(':', '-')
    time_slug = time.date().isoformat().replace('-', '')
    output_directory = f'../output/{MODEL_NAME}/{args.group}/{link_ref_slug}'

    train_normal_days = api.link_travel_time_n_preceding_normal_days(link_ref, time, N_NORMAL_DAYS)
    train_normal_days['day_type'] = cal_train.loc[train_normal_days.index.date, 'day_type'].values
    train_normal_days['link_travel_time_exp'] = train_normal_days.rolling(window=20, center=True, min_periods=1)['link_travel_time'].mean().round(1)
    test = api.link_travel_time(link_ref, time, test_end_time)
    test['day_type'] = cal_test.loc[test.index.date, 'day_type'].values
    test['link_travel_time_exp'] = test.rolling(window=20, center=True, min_periods=1)['link_travel_time'].mean().round(1)

    if len(train_normal_days) == 0 or len(test) == 0:
        continue
        
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    print('train_normal_days', len(train_normal_days))
    print('test', len(test))                
    
    model = BaselineModel()
    model.choose_tod_bins(train_normal_days)
    model.y_names = ['link_travel_time_exp']
    
    # Hyper Parameter Optimization
    if args.tune:
        K.clear_session()
        hp_train, hp_val = data_utils.split_normal_days_train_val(train_normal_days)
        X_train = model.transform(hp_train, is_training = True)
        Y_train = model.transform_y(hp_train, is_training = True)
        X_val = model.transform(hp_val, is_training = False)
        Y_val = model.transform_y(hp_val, is_training = False)
        tuner = HyperbandBatchSizeTuner(
            model.build_model,
            objective='val_loss',
            max_epochs=HYPERBAND_MAX_EPOCHS,
            executions_per_trial=EXECUTION_PER_TRIAL,
            batch_sizes=[50, 100, 500, 1000],
            seed=SEED,
            directory=output_directory,
            project_name='hyperband')

        #print(tuner.search_space_summary())

        tuner.search(X_train, Y_train,
                     epochs=20,
                     validation_data=(X_val, Y_val),
                     callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=3)])

        best_hp = tuner.get_best_hyperparameters()[0]
        
        # Final train/validation loop, using best HP to choose epochs based on Early Stopping.
        K.clear_session()
        model.build_and_train(hp_train, hp_val, best_hp, callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=10)], verbose = 0)
        hp_hist = pd.DataFrame(model.model.history.history)
        hp_hist.index.name = 'epoch'
        hp_hist.index = hp_hist.index + 1
        hp_hist.to_csv(f"{output_directory}/hp_hist_{time_slug}.csv")

        print('Epochs:', len(hp_hist))
        best_hp.values['epochs'] = len(hp_hist) - 1
    
        with open(f'{output_directory}/hyperparameters.json', 'w') as f:
            json.dump(best_hp.get_config(), f)
            
        plot_model(model.model, f'{output_directory}/model.png', show_shapes=True)
    else:
        with open(f'{output_directory}/hyperparameters.json', 'r') as f:            
            best_hp = HyperParameters.from_config(json.load(f))
    
    # Train/test loop    
    K.clear_session()
    model.build_and_train(train_normal_days, test, best_hp, verbose = 0)
    hist = pd.DataFrame(model.model.history.history)
    hist.index.name = 'epoch'
    hist.index = hist.index + 1

    train_normal_days[MODEL_NAME] = model.predict(train_normal_days)
    test[MODEL_NAME] = model.predict(test)

    train_normal_days.to_csv(f"{output_directory}/train_{time_slug}.csv")
    test.to_csv(f"{output_directory}/test_{time_slug}.csv")
    hist.to_csv(f"{output_directory}/hist_{time_slug}.csv")

    # Save visual plot of the model (for debugging)
    plot_model(model.model, f'{output_directory}/model.png', show_shapes=True)
