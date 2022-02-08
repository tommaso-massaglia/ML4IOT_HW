import tensorflow as tf
import numpy as np
import cherrypy
import json
import time
from math import exp as e
import base64
from ast import literal_eval
import zlib


def get_prob(input):
    inputexp = np.array([e(x) for x in input])
    return inputexp/sum(inputexp)*100


def predict_label(data, interpreter):
    input = np.array(data, np.float32)
    interpreter.set_tensor(input_details[0]['index'], input)
    start_inf = time.time()
    interpreter.invoke()
    my_output = tf.squeeze(interpreter.get_tensor(
        output_details[0]['index'])).numpy()

    return my_output, start_inf


class signal_preprocess:
    def __init__(self, length, stride, num_mel_bins, mel_freq, num_MFCCs):
        self.length = length
        self.stride = stride
        self.num_mel_bins = num_mel_bins
        self.mel_freq = mel_freq
        self.num_MFCCs = num_MFCCs
        self.num_frames = (mel_freq - length) // stride + 1
        self.mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, length // 2 + 1, self.mel_freq, 20, 4000)

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
        spectrogram, stft = self.get_spectrogram(tfaudio)
        mfccs = self.generate_MFFCCs(spectrogram)
        return mfccs


class commandpred(object):
    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        body = cherrypy.request.body.read()
        body = json.loads(body)

        audiofile = np.frombuffer(zlib.decompress(base64.b64decode(
            literal_eval(body.get("audiofile")))), np.float16)
        if audiofile is None:
            raise cherrypy.HTTPError(400, 'audiofile missing')
        # sampling_rate = np.array(body.get('sampling_rate'))
        # if audiofile is None:
        #     raise cherrypy.HTTPError(400, 'sampling_rate missing')

        preprocessed_file = SIG_PREPR.preprocess(audiofile, 16000)

        preds, time = predict_label(preprocessed_file, interpreter)
        probs = get_prob(preds)
        pred = COMMANDS[np.argmax(probs)]

        output = {'label': pred}

        output_json = json.dumps(output)

        #print(output_json, "Probability "+str(max(probs)))

        return output_json

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(
        model_content=open("kws_dscnn_True.tflite", "rb").read())
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    COMMANDS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
    rate = 1
    SIG_PREPR = signal_preprocess(length=640//rate,
                                  stride=320//rate,
                                  num_mel_bins=20,
                                  mel_freq=16000,
                                  num_MFCCs=10)
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(commandpred(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
