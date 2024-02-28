#Import dependencies
import serial.tools.list_ports #pyserial must be installed
import serial
from csv import writer
import keyboard
import os, time
import argparse

class recordport():
    def __init__(self,output_directory=None,output_filename=None,readrate=9600,selected_port=None):
        """ recordport class: used to record data from a port such as from an arduino computer
        output_directory = str or None, default path to where data will be saved
        output_filename = str or None, default CSV file where data will be saved in output_directory
        readrate = int, number of times comp will read port per second
        selected_port = str or None, the specific port we will listen to. Example input is 'COM3'
        """
        self.output_directory=output_directory 
        self.output_filename=output_filename 
        self.readrate=readrate  
        self.selected_port=selected_port
        self.sSerial = serial.Serial()

    def __call__(self):
        #Actions to take when object is called
        self.user_interface()

    def user_interface(self):
        #Set up directory and file for writing data
        if self.output_directory==None:
            self.set_up_output_directory()
        else:
            if not os.path.exists(self.output_directory):
                os.makedir(self.output_directory)

        if self.output_filename==None:
            self.set_up_output_file()

        # Set up port for reading
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

        self.record_port()

    def set_up_output_directory(self):
        current_time = time.strftime("%Y%m%d_%H%M%S")
        currentdir=os.getcwd()
        self.output_directory=f'{currentdir}/tmp{current_time}/'
        os.mkdir(self.output_directory)
                    
    def set_up_output_file(self):
        current_time = time.strftime("%Y%m%d_%H%M%S")
        self.output_file=f'portrecording{current_time}.csv'

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
                self.selected_port="COM"+str(correct_port)
                print(f'This is our selected port: {portList[x]}')

        self.sSerial.baudrate=self.readrate
        self.sSerial.port=self.selected_port
        self.sSerial.open()

    def record_port(self):
        print(f'Now trying to read port from {self.selected_port}')
        print(f'Type ctrl + shift + d to stop recording')
        
        self.output_file=os.path.join(self.output_directory,self.output_file) #Set final output file to its full path
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
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_directory",type=str,default=None,help="The directory where you would like to save data")
    parser.add_argument("--output_filename",type=str,default=None,help="Name of the CSV file to save data to. Please input as: yourfilename.csv")
    parser.add_argument("--readrate",type=int,default=9600,help="The rate at which port is read")
    parser.add_argument("--selected_port",default=None, help="The port we will read. Please input as: COMX where X is an int") 
    args=parser.parse_args()
    recording_port=recordport(args.output_directory,args.output_filename,args.readrate,args.selected_port)
    recording_port()