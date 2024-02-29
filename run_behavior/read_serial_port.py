#Import dependencies
import serial.tools.list_ports #pyserial must be installed
import serial
from csv import writer
import keyboard
import os, time
import argparse
import PySimpleGUI as sg
import tkinter as tk
from tkinter import *

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

    def setserial(self,index,running):
        self.sSerial[index].baudrate=self.baudlist[index]
        self.sSerial[index].port=self.portList[index]
        self.sSerial[index].open()

        while running:
            if self.sSerial[index].in_waiting:
                packet=self.sSerial[index].readline()
                decoded_packet=packet.decode('utf').rstrip('\n')

                #Write decoded packet to file
                with open(self.output_file, 'a') as fileob:
                    wro = writer(fileob)
                    wro.writerow(decoded_packet)
                    fileob.close()    

class GUI(recordport):
    def __init__(self):
        super().__init__() #Use same inputs as before
        self.root=tk.Tk()
        self.root.geometry("1000x800")
        self.root.configure(bg="black")

    def __call__(self):
        self.load_images()
        self.set_up_layout()
        self.refresh_screen()
    
    def load_images(self):
        #Get all images used by app
        self.logo = PhotoImage(file = "D:\\twophoton\\run_behavior\\gui\\logo.png") #Logo
        OnImage = PhotoImage(file = "D:\\twophoton\\run_behavior\\gui\\on.png") #On off switch
        offImage = PhotoImage(file = "D:\\twophoton\\run_behavior\\gui\\off.png")#Off switch
        sensimage = PhotoImage(file = "D:\\twophoton\\run_behavior\\gui\\sens.png") #Used for sens synce switch
        syncimage = PhotoImage(file = "D:\\twophoton\\run_behavior\\gui\\sync.png") #Used for sens synce switch
        offsso = PhotoImage(file = "D:\\twophoton\\run_behavior\\gui\\off_sso.png") #Used for sens synce switch

        # Resample images for specs of gui
        self.offImage = offImage.subsample(4, 4)
        self.logo = self.logo.subsample(5, 5)
        self.OnImage = OnImage.subsample(4, 4)
        self.sensimage = sensimage.subsample(3, 3)
        self.syncimage = syncimage.subsample(3, 3)
        self.offsso = offsso.subsample(3, 3)

    def set_up_layout(self):
        self.title = tk.Label(text="Port Reader",foreground="white", background="black")
        self.title.config(font=("Arial", 25))
        self.title.place(x=450,y=10)
        #self.logo = tk.Label(image=self.logo)
        #self.logo.place(x=10,y=0)
        self.browse_button = tk.Button(text="Set Output Folder",height=1,width=20, bg='black',fg='white',font= ('Arial 10 bold'), command=self.get_directory)
        self.browse_button.place(x=10,y=50)
        self.findportsbutton = tk.Button(self.root, text='Search for COMs', height=1, width=20, bg='black',fg='white', font= ('Arial 10 bold'), command=self.find_port)
        self.findportsbutton.place(x=10,y=90) 

    def refresh_screen(self):
        self.root.mainloop()

    def find_port(self):
        super().find_port()
        self.set_run_buttons(len(self.portList))
        return

    def get_directory(self):
        directory = tk.filedialog.askdirectory()
        string=f'Output directory: {directory}'
        self.showdir = tk.Label(text=string,foreground="white", background="black")
        self.showdir.config(font=("Arial", 15))
        self.showdir.place(x=200,y=200)
        self.root.mainloop()
        return directory

    def run_recording(self,index):
        state_oh = self.run_btn_lst[index]
        if state_oh=='on':
            self.recordbuttons[index].config(state='normal', image = self.offImage)
            self.run_btn_lst[index] = 'off'
            print(f'I, button number {index}, am off now')
        else:
            self.recordbuttons[index].config(state='normal', image = self.OnImage)
            self.run_btn_lst[index] = 'on'
            print(f'I, button number {index}, am on now')
            self.display_recording()

    def comp_type(self,index):
        state_oh = self.type_btn_lst[index]
        if state_oh=='off':
            self.typebuttons[index].config(state='normal', image = self.sensimage)
            self.type_btn_lst[index] = 'sens'
            print(f'I, button number {index}, am sens now')
        elif state_oh=='sens':
            self.typebuttons[index].config(state='normal', image = self.syncimage)
            self.type_btn_lst[index] = 'sync'
            print(f'I, button number {index}, am sync now')
        else:
            self.typebuttons[index].config(state='normal', image = self.offsso)
            self.type_btn_lst[index] = 'off'
            print(f'I, button number {index}, am off now')

    def set_run_buttons(self,n):
        self.run_btn_lst=['off'] * n
        self.type_btn_lst=['off'] * n
        self.recordbuttons=[]
        self.typebuttons=[]
        starty=300
        for index in range(n):
            #Set up the run button per comp
            btn=Button(self.root,image = self.offImage, highlightthickness = 0, bd = 0, borderwidth=0,command=lambda k=index:self.run_recording(k))
            btn.place(x=600,y=starty)
            self.recordbuttons.append(btn)

            #Set up the type button per comp
            btn=Button(self.root,image = self.offsso,highlightthickness = 0, bd = 0, borderwidth=0, command=lambda k=index:self.comp_type(k))
            btn.place(x=50,y=starty)
            self.typebuttons.append(btn)

            starty+=80

        
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