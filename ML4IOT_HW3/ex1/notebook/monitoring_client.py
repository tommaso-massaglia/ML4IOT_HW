import paho.mqtt.client as PahoMQTT
import time

class MySubscriber:
		def __init__(self, clientID):
			self.clientID = clientID
			# create an instance of paho.mqtt.client
			self._paho_mqtt = PahoMQTT.Client(clientID, False) 

			# register the callback
			self._paho_mqtt.on_connect = self.myOnConnect
			self._paho_mqtt.on_message = self.myOnMessageReceived

			self.topic = '/alerts/#'
			#self.messageBroker = 'raspberrypi.local/127.0.0.1'
			self.messageBroker = '192.168.1.131'


		def start (self):
			#manage connection to broker
			self._paho_mqtt.connect(self.messageBroker, 1883)
			self._paho_mqtt.loop_start()
			# subscribe for a topic
			self._paho_mqtt.subscribe(self.topic, 2)

		def stop (self):
			self._paho_mqtt.unsubscribe(self.topic)
			self._paho_mqtt.loop_stop()
			self._paho_mqtt.disconnect()

		def myOnConnect (self, paho_mqtt, userdata, flags, rc):
			print ("Connected to %s with result code: %d" % (self.messageBroker, rc))

		def myOnMessageReceived (self, paho_mqtt , userdata, msg):
			# A new message is received
			print (msg.payload.decode("utf-8"))

if __name__ == "__main__":
	test = MySubscriber("MySubscriber 1")
	test.start()
	for i in range(100):
		time.sleep(1)
  
	test.stop()
