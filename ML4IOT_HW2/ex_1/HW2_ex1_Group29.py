import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow.keras.layers as layers
import pathlib
import IPython
import tensorflow_model_optimization as tfmot
import zlib

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model version')
args = parser.parse_args()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

# Select only columns with humidity and temperature
column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns]

# Split the dataset into train test and val
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

# Extract the mean and standard deviation
MEAN = train_data.mean(axis=0)
STD = train_data.std(axis=0)

train_df = train_data
val_df = val_data 
test_df = test_data

#The window generator class was taken from tensorflow tutorial on Time Series and adapted
#for the purposes of the homework

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)
            
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2))
        self.count = self.add_weight('count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
            error = tf.abs(y_pred - y_true)
            error = tf.reduce_mean(error, axis=0)
            error = tf.reduce_mean(error, axis=0)
            self.total.assign_add(error)
            self.count.assign_add(1,)
        
    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return
    
    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        
        return result
    
def compile_and_fit(model, window, epochs, patience=2):
        
    model.compile(loss=tf.losses.MeanSquaredError(), \
        metrics=[ks.metrics.Accuracy(), 
                 MultiOutputMAE()
                 ], \
        optimizer=tf.optimizers.Adam())
    
    history = model.fit(x=window.train, \
                        validation_data=window.val, \
                        batch_size=32, epochs=epochs, \
                        callbacks=tfmot.sparsity.keras.UpdatePruningStep(),
                        verbose=1)
    return history


def save_model(model, model_name, window_width, out_steps, num_features, other_notes=""):
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, window_width, num_features],
                                                                  tf.float32))
    model.save("saved_models//" + model_name +other_notes, signatures=concrete_func)

epochs = 0

if args.version == "a":
    epochs = 20
    OUT_STEPS = 3
    window_width = 6
    label_columns = ["T (degC)", "rh (%)"]
    num_features = len(label_columns)

    input_shape = (window_width, num_features)
    
    window = WindowGenerator(input_width=window_width, label_width=OUT_STEPS,
                     shift=OUT_STEPS, label_columns=label_columns)

    CNNmodel = ks.Sequential([
        ks.Input(shape=(input_shape)),
        layers.Conv1D(filters=15, kernel_size=3, activation="relu"),
        layers.Flatten(input_shape=input_shape),
        layers.Dense(units=num_features*OUT_STEPS),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    model = CNNmodel

    end_step = np.ceil(len(data) / 32).astype(np.int32) * 20

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                final_sparsity=0.91,
                                                                begin_step=end_step*0.1,
                                                                end_step=end_step)
    }
    
if args.version == "b":
    epochs = 12
    OUT_STEPS = 9
    window_width = 6
    label_columns = ["T (degC)", "rh (%)"]
    num_features = len(label_columns)

    input_shape = (window_width, num_features)
    
    window = WindowGenerator(input_width=window_width, label_width=OUT_STEPS,
                     shift=OUT_STEPS, label_columns=label_columns)

    CNNmodel = ks.Sequential([
        ks.Input(shape=(input_shape)),
        layers.Conv1D(filters=24, kernel_size=4, activation="relu"),
        layers.Flatten(input_shape=input_shape),
        layers.Dense(units=num_features*OUT_STEPS),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    model = CNNmodel

    end_step = np.ceil(len(data) / 32).astype(np.int32) * 20

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                final_sparsity=0.91,
                                                                begin_step=end_step*0.25,
                                                                end_step=end_step)
    }
    
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

model_for_pruning = prune_low_magnitude(model, **pruning_params)

compile_and_fit(model_for_pruning, window, epochs)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

save_model(model_for_export, args.version, window_width, OUT_STEPS, num_features, other_notes="_ex1_29")

model_path = os.curdir+"/saved_models/"+args.version+"_ex1_29"

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


quant_model = converter.convert()

with open(f'Group29_th_{args.version}.tflite.zlib','wb') as f:
    compressed_data = zlib.compress(quant_model, 8)
    f.write(compressed_data)