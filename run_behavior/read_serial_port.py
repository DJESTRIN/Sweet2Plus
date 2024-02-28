#Import dependencies
import serial.tools.list_ports #pyserial must be installed
from csv import writer
import keyboard

class recordport():
    def __init__(self,output_directory=None,readrate=9600,selected_port=None):
        self.output_directory=output_directory
        self.readrate=9600 
        self.selected_port=selected_port
        self.sSerial = serial.Serial()

    def messages_to_user(self):
        if self.selected_port != None:
            print(f'You have manually entered {self.selected_port} as the device of interest')
            print(f'We will read this device at a rate of {self.readrate} per second')
            self.sSerial.baudrate=self.readrate
            self.sSerial.port=self.selected_port
            self.sSerial.open()
        else:
            print(f'No device selected, lets find the device you are interested in....')
            self.find_port()
            print(f'We will read this device at a rate of {self.readrate} per second')

    def set_up_output_file():
        self.output_file=

    def find_port(self):
        ports = serial.tools.list_ports.comports()
        portList = []
        print('Here is a list of all available ports:')
        for port_oh in ports:
            portList.append(str(port_oh))
            print(str(port_oh))

        correct_port = input("Please input the correct COM value as an integer. For example, if COM27 is the computer of interest, input the int 27. Go!:")

        for x in range(0,len(portList)):
            if portList[x].startswith("COM"+str(correct_port)):
                portVar="COM"+str(correct_port)
                print(f'This is our selected port: {portList[x]}')

        self.sSerial.baudrate=self.readrate
        self.sSerial.port=portVar
        self.sSerial.open()


    def record_port():
        running=True
        while running:
            if sSerial.in_waiting:
                packet=sSerial.readline()
                decoded_packet=packet.decode('utf').rstrip('\n')
                print(decoded_packet) #Print so user can see whats being saved

                #Write decoded packet to file
                with open(self.output_file, 'a') as fileob:
                    wro = writer(fileob)
                    wro.writerow(decoded_packet)
                    fileob.close()
            
            if keyboard.is_pressed("ctrl+shift+d"):
                running=False
                break
            
if __name__=='__main__':
    #output_directory
    #otuput_filename
    #readrate
    #portnumber 
    recording_port=recordport()







