import tensorflow_model_optimization as tfmot
import zlib
from tensorflow.keras import layers
from tensorflow import keras as ks
import os
import numpy as np
import pathlib
import tensorflow as tf
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=False,
                    help='model version to train')
args = parser.parse_args()


DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')

data_dir = pathlib.Path(DATASET_PATH)

np.random.seed(42)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

test_files = []
val_files = []
train_files = []

with open("kws_test_split.txt", "r") as file:
    for line in file.readlines():
        test_files.append(tf.convert_to_tensor(line.replace(
            "\n", "").replace("./", "").replace("/", "\\")))
with open("kws_val_split.txt", "r") as file:
    for line in file.readlines():
        val_files.append(tf.convert_to_tensor(line.replace(
            "\n", "").replace("./", "").replace("/", "\\")))
with open("kws_train_split.txt", "r") as file:
    for line in file.readlines():
        train_files.append(tf.convert_to_tensor(line.replace(
            "\n", "").replace("./", "").replace("/", "\\")))


test_files = tf.convert_to_tensor(test_files)
train_files = tf.convert_to_tensor(train_files)
val_files = tf.convert_to_tensor(val_files)


class signalGenerator:
    def __init__(self, keywords, sampling_rate, frame_length, frame_step, num_mel_bins, mel_freq, num_MFCCs, mfcc,
                 test_df=test_files, train_df=train_files, val_df=val_files):
        self.keywords = keywords
        self.sampling_rate = sampling_rate
        self.frame_length = int(frame_length*sampling_rate*1e-3)
        self.frame_step = int(frame_step*sampling_rate*1e-3)
        self.num_mel_bins = num_mel_bins
        self.mel_freq = mel_freq
        self.num_MFCCs = num_MFCCs
        self.mfcc = mfcc
        self.test_df = test_df
        self.train_df = train_df
        self.val_df = val_df

    def create_labels(self, file_path):
        parts = tf.strings.split(input=file_path, sep=os.path.sep)
        return tf.argmax(parts[-2] == self.keywords)

    def readandpad(self, file_path):
        file = tf.io.read_file(file_path)
        tfaudio, rate = tf.audio.decode_wav(file, desired_channels=1)
        tfaudio = tf.squeeze(tfaudio, 1)
        zero_padding = tf.zeros([rate] - tf.shape(tfaudio), dtype=tf.float32)
        waveform = tf.cast(tfaudio, dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)

        return equal_length

    def get_spectrogram(self, tfaudio):
        stft = tf.signal.stft(
            tfaudio, frame_length=self.frame_length, frame_step=self.frame_step)
        spectrogram = tf.abs(stft)

        return spectrogram, stft

    def generate_MFFCCs(self, spectrogram):
        self.mel_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins, spectrogram.shape[-1],
                                                                self.mel_freq, 20, 4000)

        mel_spectrogram = tf.tensordot(spectrogram, self.mel_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrogram)[..., :self.num_MFCCs]

        return mfccs

    def preprocess(self, file_path):
        tfaudio = self.readandpad(file_path)
        label = self.create_labels(file_path)
        spectrogram, stft = self.get_spectrogram(tfaudio)
        mfccs = self.generate_MFFCCs(spectrogram)
        if self.mfcc:
            return mfccs[..., tf.newaxis], label
        else:
            return tf.image.resize(spectrogram[..., tf.newaxis], ([32, 32])), label

    def make_dataset(self, pathslist, train):
        ds = tf.data.Dataset.from_tensor_slices(pathslist)
        ds = ds.map(map_func=self.preprocess)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, True)

    @property
    def val(self):
        return self.make_dataset(self.val_df, False)

    @property
    def test(self):
        return self.make_dataset(self.test_df, False)

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


def compile_and_fit(model, signalGenerator, patience=2):

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                  optimizer=tf.optimizers.Adam())

    history = model.fit(x=signalGenerator.train,
                        validation_data=signalGenerator.val,
                        batch_size=32, epochs=20,
                        callbacks=[
                            # early_stopping,
                            tfmot.sparsity.keras.UpdatePruningStep()
                        ], \
                        verbose=1)
    return history


def save_model(model, model_name, shape_mfccs, other_notes=""):
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, shape_mfccs[0], shape_mfccs[1], 1],
                                                                  tf.float32))
    model.save("saved_models//" + model_name +
               other_notes, signatures=concrete_func)


generator_mfccs = signalGenerator(commands,
                                  sampling_rate=16000,
                                  frame_length=40,
                                  frame_step=20,
                                  num_mel_bins=50,
                                  mel_freq=16000,
                                  num_MFCCs=10,
                                  mfcc=True)
shape_mfccs = generator_mfccs.example[0][0].shape

end_step = np.ceil(len(train_files) / 32).astype(np.int32) * 20

if args.version == "a":
    DSCNNmodel_mfccs = ks.Sequential([
        ks.Input(shape=(shape_mfccs)),
        layers.Conv2D(filters=256, kernel_size=[
                      3, 3], strides=[2, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[
                               1, 1], use_bias=False),
        layers.Conv2D(filters=256, kernel_size=[
                      1, 1], strides=[1, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[
                               1, 1], use_bias=False),
        layers.Conv2D(filters=128, kernel_size=[
                      1, 1], strides=[1, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(commands))
    ])

    model = DSCNNmodel_mfccs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                 final_sparsity=0.5,
                                                                 begin_step=end_step/2,
                                                                 end_step=end_step)
    }

if args.version == "b":
    DSCNNmodel_mfccs = ks.Sequential([
        ks.Input(shape=(shape_mfccs)),
        layers.Conv2D(filters=128, kernel_size=[
                      3, 3], strides=[2, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[
                               1, 1], use_bias=False),
        layers.Conv2D(filters=128, kernel_size=[
                      1, 1], strides=[1, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[
                               1, 1], use_bias=False),
        layers.Conv2D(filters=128, kernel_size=[
                      1, 1], strides=[1, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(commands))
    ])

    model = DSCNNmodel_mfccs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                 final_sparsity=0.2,
                                                                 begin_step=end_step/2,
                                                                 end_step=end_step)
    }

if args.version == "c":
    DSCNNmodel_mfccs = ks.Sequential([
        ks.Input(shape=(shape_mfccs)),
        layers.Conv2D(filters=128, kernel_size=[
                      3, 3], strides=[2, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[
                               1, 1], use_bias=False),
        layers.Conv2D(filters=100, kernel_size=[
                      1, 1], strides=[1, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[
                               1, 1], use_bias=False),
        layers.Conv2D(filters=90, kernel_size=[
                      1, 1], strides=[1, 1], use_bias=False),
        layers.BatchNormalization(momentum=0.1),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(commands))
    ])

    model = DSCNNmodel_mfccs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                 final_sparsity=0.5,
                                                                 begin_step=end_step/2,
                                                                 end_step=end_step)
    }

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

model_for_pruning = prune_low_magnitude(model, **pruning_params)

compile_and_fit(model, generator_mfccs)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

save_model(model, "29", shape_mfccs, f"_{args.version}")

model_path = os.curdir+"/saved_models/29_"+args.version

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quant_model = converter.convert()

with open(f'Group29_kws_{args.version}.tflite.zlib', 'wb') as f:
    compressed_data = zlib.compress(quant_model, 3)
    f.write(compressed_data)
