import requests
import pathlib
import numpy as np
from scipy.io import wavfile
import os
import tensorflow as tf
from math import exp as e
import time
import sys
import json
import base64
import zlib

from subprocess import Popen
Popen('sudo sh -c "echo performance >'
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
      shell=True).wait()


def get_prob(input):
    inputexp = np.array([e(x) for x in input])
    return inputexp/sum(inputexp)*100


def predict_label(data, interpreter):
    input = np.array(data, np.float32)
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    my_output = tf.squeeze(interpreter.get_tensor(
        output_details[0]['index'])).numpy()

    return my_output


class signal_preprocess:
    def __init__(self, length, stride, num_mel_bins, mel_freq, num_MFCCs):
        self.length = length
        self.stride = stride
        self.num_mel_bins = num_mel_bins
        self.mel_freq = mel_freq
        self.num_MFCCs = num_MFCCs
        self.num_frames = (mel_freq - length) // stride + 1
        self.mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, length // 2 + 1, self.mel_freq, 20, 2666)

    def readandpad(self, audio):
        max_nb_bit = float(2 ** (16 - 1))
        audio = audio / (max_nb_bit + 1)
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        waveform = tf.cast(audio, dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)

        return equal_length

    def get_spectrogram(self, tfaudio):
        stft = tf.signal.stft(tfaudio, frame_length=self.length,
                              frame_step=self.stride, fft_length=self.length)
        spectrogram = tf.abs(stft)

        return spectrogram, stft

    def generate_MFFCCs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram, self.mel_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrogram)[..., :self.num_MFCCs]
        mfccs = tf.reshape(mfccs, [1, self.num_frames, self.num_MFCCs, 1])
        return mfccs

    def preprocess(self, audio, sampling_rate):
        tfaudio = self.readandpad(audio)
        tfaudio = tfaudio[::sampling_rate // self.mel_freq]
        # if sampling_rate!=self.mel_freq:
        #     tfaudio = signal.resample_poly(tfaudio, 1, sampling_rate // self.mel_freq)
        #     tfaudio = tf.convert_to_tensor(tfaudio, dtype=tf.float32)
        spectrogram, stft = self.get_spectrogram(tfaudio)
        mfccs = self.generate_MFFCCs(spectrogram)
        return mfccs


url = 'http://192.168.183.87:8080'

# Initialize the interpreter
interpreter = tf.lite.Interpreter(
    model_content=open("kws_dscnn_True.tflite", "rb").read())
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
commands = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
rate = 3

sig_prepr = signal_preprocess(length=640//rate,
                              stride=320//rate,
                              num_mel_bins=20,
                              mel_freq=16000//rate,
                              num_MFCCs=10)

tot_latency = []
comm_cost = []
totfiles = 0
correct = 0

for el in os.listdir("kws_test_split"):
    filepath = pathlib.Path(f"kws_test_split/{el}").as_posix()
    start = time.time()
    sampling_rate, audiofile = wavfile.read(filepath)
    mfccs = sig_prepr.preprocess(audiofile, sampling_rate)
    output = predict_label(mfccs, interpreter)
    probs = get_prob(output)
    pred = commands[np.argmax(probs)]
    true = el.split("-")[0]
    totfiles += 1
    slowlabel = ""
    end = time.time()

    if max(probs) < 86:
        compressed_audio = str(base64.b64encode(
            zlib.compress(np.array(audiofile, dtype=np.float16), 3)))
        end = time.time()
        body = {'audiofile': compressed_audio}
        #print(f"called slow_service at file {totfiles}")
        r = requests.put(url, json=body)
        comm_cost.append(sys.getsizeof(compressed_audio))

        if r.status_code == 200:
            body = r.json()
            comm_cost.append(sys.getsizeof(body["label"]))
            slowlabel = body["label"]
        else:
            print('Error:', r.status_code)

    correct += 1 if pred == true or slowlabel == true else 0
    tot_latency.append(end - start)

print(f"Accuracy = {correct/totfiles*100:.2f} %")

print('Total Latency {:.2f}ms'.format(np.mean(tot_latency)*1000.))
print('Total Communication Cost {:.3f} MB'.format(
    np.sum(comm_cost)/8/1024/1024))
