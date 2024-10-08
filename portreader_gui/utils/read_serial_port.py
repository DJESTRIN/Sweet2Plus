#Import dependencies
import serial.tools.list_ports #pyserial must be installed
import serial
from csv import writer
import keyboard
import os, time
import argparse
import tkinter as tk
from tkinter import *
from threading import Thread
from tkinter import filedialog
import ipdb

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
        self.selected_port=selected_port
        self.sSeriallist = []
        self.baudlist = [str(readrate)]
                    
    def set_up_output_file(self):
        self.output_files=[]
        for index in range(len(self.portList)):
            comoh=self.portList[index]
            comoh=comoh.split(' ')
            comoh=comoh[0]
            current_time = time.strftime("%Y%m%d_%H%M%S")
            self.output_files.append(f'{comoh}_{current_time}.csv')

    def find_port(self):
        ports = serial.tools.list_ports.comports()
        portList = []
        print('Here is a list of all available ports:')
        for port_oh in ports:
            portList.append(str(port_oh))
            print(str(port_oh))
        self.portList=portList

    def record_port(self,index,running):
        self.sSeriallist[index].baudrate=int(self.baudlist[index])
        comoh=self.portList[index]
        comoh = comoh.split(' ')
        comoh = comoh[0]
        self.sSeriallist[index].port=comoh
        # if running:
        #     self.sSeriallist[index].open()
        # else:
        #     self.sSeriallist[index].close()
   

class GUI(recordport):
    def __init__(self,wd,testmode=False):
        super().__init__() #Use same inputs as before
        self.wd=wd
        self.testmode=testmode
        self.root=tk.Tk()
        self.root.geometry("1500x800+300+100")
        self.root.configure(bg="black")
        self.root.overrideredirect(True)
        self.guirunning=True
        exit_button = Button(self.root, text="Exit" ,height=1,width=10, bg='black',fg='white',font= ('Arial 10 bold'), command=self.ending_gui)
        exit_button.place(x=1400,y=10)

    def __call__(self):
        self.load_images()
        self.set_up_layout()
        self.refresh_screen()
    
    def load_images(self):
        #Get all images used by app
        self.logo = PhotoImage(file = f'{self.wd}\gui\logo.png') #Logo
        OnImage = PhotoImage(file = f'{self.wd}\gui\on.png') #On off switch
        offImage = PhotoImage(file = f'{self.wd}\gui\off.png')#Off switch
        sensimage = PhotoImage(file = f'{self.wd}\gui\sens.png') #Used for sens synce switch
        syncimage = PhotoImage(file = f'{self.wd}\gui\sync.png') #Used for sens synce switch
        offsso = PhotoImage(file = f'{self.wd}\gui\off_sso.png') #Used for sens synce switch

        # Resample images for specs of gui
        self.offImage = offImage.subsample(4, 4)
        self.logo = self.logo.subsample(5, 5)
        self.OnImage = OnImage.subsample(4, 4)
        self.sensimage = sensimage.subsample(3, 4)
        self.syncimage = syncimage.subsample(3, 4)
        self.offsso = offsso.subsample(3, 4)

    def set_up_layout(self):
        self.title = tk.Label(text="Port Reader",foreground="white", background="black")
        self.title.config(font=('Arial', '25', 'bold'))
        self.title.pack()
        self.browse_button = tk.Button(text="Set Output Folder",height=1,width=20, bg='black',fg='white',font= ('Arial', '10', 'bold'), command=self.get_directory)
        self.browse_button.place(x=10,y=50)
        self.findportsbutton = tk.Button(self.root, text='Search for COMs', height=1, width=20, bg='black',fg='white', font= ('Arial', '10', 'bold'), command=self.find_port)
        self.findportsbutton.place(x=10,y=90) 

    def refresh_screen(self):
        self.root.mainloop()

    def find_port(self):
        super().find_port()
        self.set_run_buttons(len(self.portList))
        return

    def get_directory(self):
        directory = filedialog.askdirectory(initialdir=self.wd)
        string=f'Output directory: {directory}'
        self.showdir = tk.Label(text=string,foreground="white", background="black")
        self.showdir.config(font=("Arial", 15))
        self.showdir.place(x=200,y=50)
        self.output_directory=directory
        self.refresh_screen()

    def run_recording(self,index):
        state_oh = self.run_btn_lst[index]
        if state_oh=='on':
            self.recordbuttons[index].config(state='normal', image = self.offImage)
            self.run_btn_lst[index] = 'off'
            self.record_port(index,False)
        else:
            self.recordbuttons[index].config(state='normal', image = self.OnImage)
            self.run_btn_lst[index] = 'on'
            self.record_port(index,True)
        self.wait_for_events()

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

    def kill_switch(self):
        # Turn off all recordings
        for index in range(len(self.portList)):
            self.recordbuttons[index].config(state='normal', image = self.offImage)
            self.run_btn_lst[index] = 'off'

    def set_run_buttons(self,n):
        starty=200
        # Set up subtitles to buttons
        lab=Label(self.root,text="Available COMS ",bg='black',fg='white',font=('Arial', 20, 'bold'))
        lab.place(x=10,y=starty-50)
        lab=Label(self.root,text="Filename",bg='black',fg='white',font=('Arial', 15, 'bold'))
        lab.place(x=1015,y=starty-50)
        lab=Label(self.root,text="Baud rate",bg='black',fg='white',font=('Arial', 15, 'bold'))
        lab.place(x=800,y=starty-50)
        lab=Label(self.root,text="COM Type",bg='black',fg='white',font=('Arial', 15, 'bold'))
        lab.place(x=1220,y=starty-50)
        lab=Label(self.root,text="Recording",bg='black',fg='white',font=('Arial', 15, 'bold'))
        lab.place(x=1380,y=starty-50)

        self.run_btn_lst=['off'] * n
        self.type_btn_lst=['off'] * n
        self.baudlist = ['115200'] * n
        self.recordbuttons=[]
        self.typebuttons=[]
        self.sSeriallist=[]
        self.entrybuttons=[]
        self.tempoutput=[]
        self.finalbauds=[]

        self.set_up_output_file()

        for index in range(n):
            #set up serial
            if self.testmode:
                soh = serial.Serial()
                soh.baudrate=115200
                portname=self.portList[index]
                portname=portname.split(' ')
                portname=portname[0]
                soh.port=portname
                self.sSeriallist.append(soh) 
            else:
                try:
                    soh = serial.Serial()
                    soh.baudrate=115200
                    portname=self.portList[index]
                    portname=portname.split(' ')
                    portname=portname[0]
                    soh.port=portname
                    soh.open()
                    self.sSeriallist.append(soh)
                except:
                    continue

            #set up label
            stringoh=self.portList[index]
            lb=Label(self.root,text=stringoh,bg='black',fg='white',font=('Arial', 15, 'bold'))
            lb.place(x=10,y=starty+5)

            #Set up the run button per comp
            btn=Button(self.root,image = self.offImage, highlightthickness = 0, bd = 0, borderwidth=0,command=lambda k=index:self.run_recording(k))
            btn.place(x=1400,y=starty)
            self.recordbuttons.append(btn)

            #Set up the type button per comp
            btn=Button(self.root,image = self.offsso,highlightthickness = 0, bd = 0, borderwidth=0, command=lambda k=index:self.comp_type(k))
            btn.place(x=1200,y=starty)
            self.typebuttons.append(btn)

            # Set entry text for save file
            self.tempoutput.append(tk.StringVar())
            textBox = tk.Entry(self.root,width=35,textvariable=self.tempoutput[index]) 
            textBox.insert(0, self.output_files[index])
            textBox.place(x=950,y=starty+10)
            self.entrybuttons.append(textBox) 

            # Set entry text for baudrate
            self.finalbauds.append(tk.StringVar())
            textBox = tk.Entry(self.root,width=15,textvariable=self.finalbauds[index]) 
            textBox.insert(0, self.baudlist[index])
            textBox.place(x=800,y=starty+10)
            self.entrybuttons.append(textBox) 

            starty+=50

        endbtn=Button(self.root,height=1,width=20, bg='black',fg='white',font= ('Arial', '10', 'bold'),command=self.kill_switch)
        endbtn.config(text='End All Recordings')
        endbtn.place(x=1310,y=starty+100)

        filebtn=Button(self.root,height=1,width=20, bg='black',fg='white',font= ('Arial', '10', 'bold'),command=self.set_output_files)
        filebtn.config(text='Set Filenames')
        filebtn.place(x=970,y=starty+100)

        bdbtn=Button(self.root,height=1,width=12, bg='black',fg='white',font= ('Arial', '10', 'bold'),command=self.set_baud_rates)
        bdbtn.config(text='Set Baud Rate')
        bdbtn.place(x=798,y=starty+100)
    
    def set_output_files(self):
        for index in range(len(self.portList)):
            new_output = self.tempoutput[index].get()
            print(new_output)

    def set_baud_rates(self):
        for index in range(len(self.portList)):
            new_output = self.finalbauds[index].get()
            print(new_output)

    def ending_gui(self):
        self.guirunning=False
        self.root.destroy()

    def continous_sampling(self):
        while self.guirunning:
            for index in range(len(self.portList)):
                onoroff=self.run_btn_lst[index]
                if onoroff=='on':
                    if self.testmode:
                        decoded_packet=f'This is a test for port: {self.portList[index]}. \n It is working well!'
                        final_output=os.path.join(self.output_directory,self.output_files[index])
                        with open(final_output, 'a') as fileob: #OUTPUTFILE NEEDS TO BE INDEXED
                                wro = writer(fileob)
                                wro.writerow(decoded_packet)
                                print(decoded_packet)
                                fileob.close() 
                    else:
                        if self.sSeriallist[index].in_waiting:
                            packet=self.sSeriallist[index].readline()
                            decoded_packet=packet.decode('utf').rstrip('\n')

                            #Write decoded packet to file
                            final_output=os.path.join(self.output_directory,self.output_files[index])
                            with open(final_output, 'a') as fileob: #OUTPUTFILE NEEDS TO BE INDEXED
                                wro = writer(fileob)
                                wro.writerow(decoded_packet)
                                print(decoded_packet)
                                fileob.close() 

    def wait_for_events(self):
        Thread(target = self.continous_sampling).start()
        self.refresh_screen()


if __name__=='__main__':
    try:
        parser=argparse.ArgumentParser()
        parser.add_argument("--working_dir",required=True,type=str)
        args = parser.parse_args()
        workingd=args.working_dir
    except:
        workingd=os.path.join(os.getcwd(),'run_behavior')

    goh=GUI(workingd,True)
    goh()