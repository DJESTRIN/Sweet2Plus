import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from customs2p import manual_classification
from threading import Thread
import ipdb
import matplotlib.pyplot as plt
import cv2

ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class quickGUI(manual_classification):
    def __init__(self,datapath):
        super().__init__(datapath)
        self.root=ctk.CTk()
        self.root.geometry("700x750+500+100")
        self.canvas = tk.Canvas(self.root,width=1000,height=1000)
        self.canvas.pack()
        self.i=0
        self.interval=20
        self.neuron_number=0
        self.set_up_buttons()

    def show_vid(self): 
        im_oh = self.create_vids(self.neuron_number,self.i)
        im_oh *= 100
        im_oh[np.where(im_oh>255)]=255
        im_oh = Image.fromarray(im_oh.astype(np.uint8))
        self.i+=1
        if self.i>len(self.images):
            self.i=0
        self.img =  ImageTk.PhotoImage(image=im_oh,size=(800,800))
        self.canvas.create_image(500,500, anchor="c", image=self.img)
        self.root.after(self.interval,self.show_vid)

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

    def next_neuron(self,_event=None):
        self.neuron_number+=1
        self.i=0
        if self.neuron_number+1>len(self.images):
            self.neuron_number=0
    
    def previous_neuron(self,_event=None):
        self.neuron_number-=1
        self.i=0
        if self.neuron_number-1<0:
            self.neuron_number=len(self.images)
    
    def is_neuron(self,_event=None):
        self.true_classification[self.neuron_number]=1
        print('Hurray')
    
    def is_not_neuron(self,_event=None):
        self.true_classification[self.neuron_number]=0
        print('BOOO!')
    
    def __call__(self):
        super().__call__()
        self.true_classification=np.zeros(shape=(len(self.traces),1))
        self.show_vid()
        self.root.mainloop()
        self.save_data()
    
    def save_data(self):
        ipdb.set_trace()
        np.save(self.true_classification)

if __name__=='__main__':
    ev_obj=quickGUI(r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620083_M4_R1-054')
    ev_obj()