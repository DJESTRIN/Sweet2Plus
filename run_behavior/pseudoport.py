import serial.tools.list_ports #pyserial must be installed
import serial
from threading import Thread
import os
from winpty import PtyProcess
import ipdb

class fakeport():
    def __init__(self,number_of_ports):
        self.ports=[]
        for i in range(number_of_ports):
            proc = PtyProcess.spawn('python')
            portname = os.ttyname(proc)
            #portname=f'COM{i}'
            self.ports[i]=serial.Serial(portname,9600)
        self.run()

    def continous_writing(self):
        for i in range(len(self.ports)):
            self.ports[i].write(b'Hey, how are you?')
    
    def run(self):
        Thread(target=self.continous_writing).start()


if __name__=='__main__':
    fports=fakeport(5)
