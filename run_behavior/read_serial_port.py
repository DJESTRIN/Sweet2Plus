#Import dependencies
import serial.tools.list_ports #pyserial must be installed
import serial
from csv import writer
import keyboard
import os, time
import argparse
import PySimpleGUI as sg
import tkinter as tk

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
        self.portList=portList
        # correct_port = input("Please input the correct COM value as an integer. For example, if COM27 is the computer of interest, input the int 27. Go!:")

        # for x in range(0,len(portList)):
        #     if portList[x].startswith("COM"+str(correct_port)):
        #         self.selected_port="COM"+str(correct_port)
        #         print(f'This is our selected port: {portList[x]}')

        # self.sSerial.baudrate=self.readrate
        # self.sSerial.port=self.selected_port
        # self.sSerial.open()

    def record_port(self):
        print(f'Now trying to read port from {self.selected_port}')
        print(f'Type ctrl + shift + d to stop recording')
        
        self.output_file=os.path.join(self.output_directory,self.output_file) #Set final output file to its full path
        running=True
        while running:
            if self.sSerial.in_waiting:
                packet=self.sSerial.readline()
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

class GUI(recordport):
    def __init__(self):
        super().__init__() #Use same inputs as before
        self.root=tk.Tk()
        self.root.geometry("1000x1000")
        self.root.configure(bg="black")
        self.title = tk.Label(text="Port Reader",foreground="white", background="black")
        self.title.config(font=("Arial", 25))
        self.title.pack()
        #self.root.mainloop()
    
    def __call__(self):
        self.set_up_buttons()
        #self.file_save_entries()
        self.root.mainloop()

    def refresh_screen(self):
        self.root.mainloop()

    def set_up_buttons(self):
        self.browse_button = tk.Button(text="Set Output Folder",
                                       height=1,width=20, bg='black',fg='white', 
                                       font= ('Arial 10 bold italic'),
                                       command=self.get_directory)
        self.browse_button.place(x=10,y=50)
        self.findportsbutton = tk.Button(self.root, text='Search for COMs', 
                                         height=1, width=20, bg='black',fg='white', 
                                         font= ('Arial 10 bold italic'),
                                         command=self.find_port)
        self.findportsbutton.place(x=10,y=90) 

    def get_directory(self):
        directory = tk.filedialog.askdirectory()
        string=f'Output directory: {directory}'
        self.showdir = tk.Label(text=string,foreground="white", background="black")
        self.showdir.config(font=("Arial", 15))
        self.showdir.place(x=200,y=45)
        self.root.mainloop()
        return directory
    
    def find_port(self):
        super().find_port()
        #Set subtitle to display all the coms
        self.subtitle=tk.Label(text="Here are the following COMS: ", fg="white",bg="black")
        self.subtitle.config(font=("Arial", 15))
        start=120
        self.subtitle.place(x=10, y=start)

        self.subtitle2=tk.Label(text="Select Arduino type", fg="white",bg="black")
        self.subtitle2.config(font=("Arial", 15))
        self.subtitle2.place(x=500, y=start)

        self.subtitle3=tk.Label(text="Set Baud Rate", fg="white",bg="black")
        self.subtitle3.config(font=("Arial", 15))
        self.subtitle3.place(x=700, y=start)

        for i,port in enumerate(self.portList):
            start+=30
            # Show the COM name
            self.showdir = tk.Label(text=port,foreground="white", background="black")
            self.showdir.config(font=("Arial", 15))
            self.showdir.place(x=10,y=start)

            #Generate radio button options
            self.rb=tk.Checkbutton(self.root,text="sync",fg='white',bg='black',selectcolor='black')
            self.rb.config(font=("Arial", 15))
            self.rb.place(x=500,y=start-3)
            self.rb=tk.Checkbutton(self.root,text="sensor",fg='white',bg='black',selectcolor='black')
            self.rb.config(font=("Arial", 15))
            self.rb.place(x=580,y=start-3)

            #Generate baud rate
            self.textBox = tk.Entry(self.root) 
            self.textBox.place(x=700,y=start+2)
            self.textBox.insert(0, '9600') 

            #Generate filename options to save csv file to
            self.textBox = tk.Entry(self.root) 
            self.textBox.place(x=850,y=start+2)
            example_filename=(f'Arduino_{i}.csv')
            self.textBox.insert(0, example_filename) 

        #Set the run button
        self.run_button = tk.Button(self.root, text="RUN",
                                       height=3,width=20, bg='red',fg='black', 
                                       font= ('Arial 10 bold'),
                                       command=self.run_all)
        self.run_button.place(relx=0.9,rely=0.9,anchor='center')

        self.refresh_screen()

    def run_all(self):
        print('Running')


        
if __name__=='__main__':
    goh=GUI()
    goh()
    # parser=argparse.ArgumentParser()
    # parser.add_argument("--output_directory",type=str,default=None,help="The directory where you would like to save data")
    # parser.add_argument("--output_filename",type=str,default=None,help="Name of the CSV file to save data to. Please input as: yourfilename.csv")
    # parser.add_argument("--readrate",type=int,default=9600,help="The rate at which port is read")
    # parser.add_argument("--selected_port",default=None, help="The port we will read. Please input as: COMX where X is an int") 
    # args=parser.parse_args()
    # recording_port=recordport(args.output_directory,args.output_filename,args.readrate,args.selected_port)
    # recording_port()

    # Look for location of file
    # prompt name asking for filename 
    # two serial pulls at same time. 
    # Read two ports at the same time. COM4 and COM9 
    # 115200
    # 9600
    # Two different files. 