import cherrypy
import json
import base64
from ast import literal_eval
import os
import paho.mqtt.client as PahoMQTT
import numpy as np
import adafruit_dht
import zlib
import board
import tensorflow as tf
import datetime
import time

class MyPublisher:

    def __init__(self, clientID):
        self.clientID = clientID

        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(self.clientID, False)
        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect

        #self.messageBroker = 'mqtt.eclipseprojects.io'
        self.messageBroker = '0.0.0.0'

    def start(self):
        # manage connection to broker
        self._paho_mqtt.connect(self.messageBroker, 1883)
        self._paho_mqtt.loop_start()

    def stop(self):
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()

    def myPublish(self, topic, message):
        # publish a message with a certain topic
        self._paho_mqtt.publish(topic, message, 2)

    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" %
              (self.messageBroker, rc))
        pass


class temperature_alerter:
    def predict_temp(self, data, interpreter):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input = np.expand_dims(np.array(data, np.float32), 0)
        interpreter.set_tensor(input_details[0]['index'], input)
        interpreter.invoke()
        my_output = tf.squeeze(interpreter.get_tensor(
            output_details[0]['index'])).numpy()

        return my_output

    def __init__(self):
        self.test = MyPublisher("MyPublisher")

    def main(self, model, DHT, tthres, hthres, runtime=60):
        self.test.start()
        print("Service Started")
        modelname = f"models/{model}.tflite.zlib"

        with open(modelname, 'rb') as f:
            decompressed_data = f.read()
            decompressed_data = zlib.decompress(decompressed_data)
        # Initialize the interpreter
        interpreter = tf.lite.Interpreter(model_content=decompressed_data)
        interpreter.allocate_tensors()

        res = []

        for i in range(6):
            try:
                temperature = DHT.temperature
                humidity = DHT.humidity
                res.append((temperature, humidity))
                time.sleep(1)
            except RuntimeError:
                self.test.myPublish('/errors', "DHT11 error")
                res.append(res[0])
                return False

        
        for j in range(runtime):
            try:
                temperature = DHT.temperature
                humidity = DHT.humidity
                res.append((temperature, humidity))
                time.sleep(1)
                ptemp, prh = self.predict_temp(res[j:6+j], interpreter)
                ptemp, prh = float(ptemp), float(prh)
                ttemp, trh = float(res[6+j][0]), float(res[6+j][1])

                if abs(ptemp-ttemp) > tthres:
                    response = f"({str(datetime.datetime.now())}) Temperature Alert: Predicted={round(ptemp, 2)}°C Actual={round(ttemp,2)}°C"
                    print(response)
                    self.test.myPublish('/alerts/temp', response)

                if abs(prh-trh) > hthres:
                    response = f"({str(datetime.datetime.now())}) Humidity Alert: Predicted={round(prh,2)}% Actual={round(trh, 2)}%"
                    print(response)
                    self.test.myPublish('/alerts/humid', response)

                results = {
                    "predicted T": float(ptemp),
                    "predicted H": float(prh),
                    "true T": float(ttemp),
                    "true H": float(trh)
                }
                self.test.myPublish('/regular', json.dumps(results))

            except RuntimeError:
                self.test.myPublish('/errors', "DHT11 error")
                res.append(res[0])
                return False
                
        self.test.stop()
        return True

class modelregistry(object):
    exposed = True

    def PUT(self, *path, **query):
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')
        
        body = cherrypy.request.body.read()
        body = json.loads(body)
        model = body.get('model')
        tthres = body.get('tthres')
        hthres = body.get('hthres')
        duration = body.get('duration')

        if len(os.listdir("models"))<2:
            raise cherrypy.HTTPError(400, 'Not enough models')            

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        if model is None:
            model = "CNN"
        if tthres is None:
            tthres = 0.1
        if hthres is None:
            hthres = 0.2
        if duration is None:
            duration = 60

        TALERTER.main(model, DHT, tthres, hthres, duration)
        return json.dumps("Succesful Operation")
        
        
    def GET(self, *path, **query):
        output_json = json.dumps(os.listdir("models"))

        return output_json

    def POST(self, *path, **query):

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        body = cherrypy.request.body.read()
        body = json.loads(body)

        model = body.get('model')
        modelname = body.get('modelname')

        if model is None:
            raise cherrypy.HTTPError(400, 'model missing')

        if modelname is None:
            raise cherrypy.HTTPError(400, 'modelname missing')

        with open(f"models/{modelname}.tflite.zlib", "wb") as f:
            f.write(base64.b64decode(literal_eval(model)))

        output = f"Model {modelname} successfully written."

        output_json = json.dumps(output)

        return output_json

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    TALERTER = temperature_alerter()
    DHT = adafruit_dht.DHT11(board.D18)
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(modelregistry(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()