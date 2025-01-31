import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from Sweet2Plus.core.customs2p import manual_classification
from threading import Thread
import ipdb
import matplotlib.pyplot as plt
import cv2
import os

ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class quickGUI(manual_classification):
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.7,redogui=False):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=cellthreshold)
        self.root=ctk.CTk()
        self.root.geometry("800x800+500+100")
        self.canvas = tk.Canvas(self.root,width=1000,height=1000)
        self.canvas.pack()
        self.i=0
        self.interval=5
        self.neuron_number=0
        self.intensity=10
        self.redogui=False
        self.set_up_buttons()
        self.skipgui=True

    def show_vid(self): 
        im_oh = self.create_vid_for_gui(self.neuron_number,self.i,self.intensity)
        im_oh = Image.fromarray(im_oh.astype(np.uint8))
        self.i+=1
        if self.i>len(self.corrected_images):
            self.i=0
        self.img =  ImageTk.PhotoImage(master=self.root,image=im_oh,size=(800,800))
        self.canvas.create_image(500,500, anchor="c", image=self.img)
        self.afterid=self.root.after(self.interval,self.show_vid)

    def set_up_buttons(self):
        self.main_button_1 = ctk.CTkButton(master=self.root,  text="Next ROI",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.next_neuron)
        self.main_button_1.place(relx=0.3,rely=0.9)
        self.root.bind('d',self.next_neuron)
        self.main_button_2 = ctk.CTkButton(master=self.root,  text="Previous ROI",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.previous_neuron)
        self.main_button_2.place(relx=0.1,rely=0.9)
        self.root.bind('a',self.previous_neuron)
        self.main_button_3 = ctk.CTkButton(master=self.root,  text="Is A Neuron",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.is_neuron)
        self.main_button_3.place(relx=0.5,rely=0.9)
        self.root.bind('w',self.is_neuron)
        self.main_button_4 = ctk.CTkButton(master=self.root,  text="Is Not A Neuron",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.is_not_neuron)
        self.main_button_4.place(relx=0.7,rely=0.9)
        self.root.bind('s',self.is_not_neuron)
        self.main_button_m = ctk.CTkButton(master=self.root,  text="toggle mask",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.change_mask)
        self.main_button_m.place(relx=0.6,rely=0.9)
        self.root.bind('e',self.change_mask)
        self.main_button_4 = ctk.CTkButton(master=self.root,  text="Increase Intensity",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.increase_intensity)
        self.main_button_4.place(relx=0.8,rely=0.9)
        self.root.bind('+',self.increase_intensity)
        self.main_button_4 = ctk.CTkButton(master=self.root,  text="Decrease Intensity",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.decrease_intensity)
        self.main_button_4.place(relx=0.9,rely=0.9)
        self.root.bind('-',self.decrease_intensity)
        self.main_button_close = ctk.CTkButton(master=self.root,  text="Closewindow",border_width=2, text_color=("gray10", "#DCE4EE"),command=self.close_gui)
        self.main_button_close.place(relx=0.9,rely=0.1)

    def close_gui(self):
        tk.Tk.after_cancel(self.root,self.afterid)
        self.root.destroy()

    def change_mask(self,_event=None):
        print(f'set mask is {self.include_mask}')
        if self.include_mask:
            self.include_mask=False
        else:
            self.include_mask=True

    def next_neuron(self,_event=None):
        self.neuron_number+=1
        self.i=0
        if self.neuron_number+1>len(self.traces):
            self.neuron_number=0
    
    def previous_neuron(self,_event=None):
        self.neuron_number-=1
        self.i=0
        if self.neuron_number-1<0:
            self.neuron_number=len(self.traces)-1
    
    def is_neuron(self,_event=None):
        self.true_classification[self.neuron_number]=1
        print('Hurray')
    
    def is_not_neuron(self,_event=None):
        self.true_classification[self.neuron_number]=0
        print('BOOO!')

    def increase_intensity(self):
        self.intensity+=1

    def decrease_intensity(self):
        self.intensity-=1
        if self.intensity<1:
            self.intensity=1

    def __call__(self):
        super().__call__()
        self.true_classification=np.zeros(shape=(len(self.traces),1))
        self.skip_gui()
        self.skipgui=True
        if self.skipgui:
            print('App for manually classifying ROIs as neurons was skipped.')
            print(f'Please see: {self.true_class_filename}')
        else:
            try:
                self.show_vid()
                self.root.mainloop()
                self.save_data()
            except:
                self.save_data()
            #Update trace and stat
            self.traces = self.traces[np.where(self.true_classification==1)[0]]
            self.stat = self.stat[np.where(self.true_classification==1)[0]]
    
    def save_data(self):
        np.save(self.true_class_filename ,self.true_classification)

    def skip_gui(self):
        drop_path,_=self.probability_files[0].split('iscell')
        self.true_class_filename = os.path.join(drop_path,'iscell_manualcut.npy')
        #Skip the gui if file exists
        if os.path.isfile(self.true_class_filename):
            self.skipgui=True
            self.true_classification=np.load(self.true_class_filename)
            self.traces = self.traces[np.where(self.true_classification==1)[0]]
            self.stat = self.stat[np.where(self.true_classification==1)[0]]
        else:
            self.skipgui=False

        if self.redogui:
            self.skipgui=False

if __name__=='__main__':
    ev_obj=quickGUI(r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620081_M1_R1-058',redogui=True)
    ev_obj()