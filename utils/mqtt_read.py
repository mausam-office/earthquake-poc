import logging
from datetime import datetime
from multiprocessing import Queue
import time

from paho.mqtt import client as mqtt_client
from paho.mqtt.enums import CallbackAPIVersion

from collections import deque


class MQTT:
    def __init__(self, q_maxsize: int, configs: dict, parent=None):
        assert isinstance(configs, dict), f"`configs` must be of dict type."

        self.queue = Queue(maxsize=q_maxsize)
        # self.bck_queue = Queue(maxsize=q_maxsize)
        self.q_maxsize= q_maxsize
        self.configs = configs
        self.running = False
    
    def collect_data(self):
        client = self.connect_mqtt()
        # start_time = time.time()
        client.loop_start()
        # while True:
        #     if (time.time() - start_time) >= 1:
        #         print(f"{self.queue.qsize()=}")
        #         break

        
    def connect_mqtt(self, ):
        client = mqtt_client.Client(CallbackAPIVersion.VERSION2, client_id="Earthquake Subs")
        client.username_pw_set('technology', 'Technology@2023')
        client.on_connect = self.on_mqtt_connection
        client.on_message = self.on_message
        client.connect(
            self.configs['broker'],
            self.configs['port'],
        )
        return client
    
    def on_mqtt_connection(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
        else:
            # we should always subscribe from on_connect callback to be sure
            # our subscribed is persisted across reconnections.
            client.subscribe(self.configs['topic'])
            logging.info('Connected and subscribed.')

    def on_message(self, client, userdata, message):
        # topic, payload, qos, retain
        
        value = float(message.payload.decode('utf-8').split('=')[-1])
        if self.queue.qsize() < self.q_maxsize:
            self.queue.put(value)
            dt = datetime.now()
            # print(dt, f'\t {value}')

        # elif self.queue.qsize() == self.q_maxsize:
        #     'queue is full but data stream is active'
        #     self.bck_queue.put(value)

        # else: # if lastest data is needed
        #     self.queue.get()
        #     self.queue.put(value)

    def read(self):
        data = deque()
       
        while len(data) <= 100 and not self.queue.empty():
            data.append(self.queue.get())
        return data


    