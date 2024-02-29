import tkinter as tk
from tkinter import *
import ipdb

class GUI():
    def __init__(self):
        #super().__init__() #Use same inputs as before
        self.root=tk.Tk()
        self.root.geometry("1000x800")
        self.portList=20
        self.root.configure(bg="black")

    def __call__(self):
        self.load_images()
        self.set_up_layout()
        self.set_run_buttons(self.portList)
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
        self.offImage = offImage.subsample(7, 7)
        self.logo = self.logo.subsample(5, 5)
        self.OnImage = OnImage.subsample(7, 7)
        sensimage = sensimage.subsample(7, 7)
        syncimage = syncimage.subsample(7, 7)
        offsso = offsso.subsample(7, 7)

    def set_up_layout(self):
        self.title = tk.Label(text="Port Reader",foreground="white", background="black")
        self.title.config(font=("Arial", 25))
        self.title.place(x=100,y=50)
        self.logo = tk.Label(image=self.logo)
        self.logo.place(x=10,y=0)

    def refresh_screen(self):
        self.root.mainloop()

    def switch(self,index):
        state_oh = btn_lst[index]
        if state_oh=='on':
            self.buttons[index].config(state='normal', image = self.offImage)
            btn_lst[index] = 'off'
            print(f'I, button number {index}, am off now')
        else:
            self.buttons[index].config(state='normal', image = self.OnImage)
            btn_lst[index] = 'on'
            print(f'I, button number {index}, am on now')

    def set_run_buttons(self,n):
        global btn_lst
        btn_lst=['off'] * n
        self.buttons=[]
        starty=0
        for index,state in enumerate(btn_lst):
            btn=Button(self.root,image = self.offImage, command=lambda k=index:self.switch(k))
            btn.place(x=100,y=starty)
            self.buttons.append(btn)
            starty+=20

if __name__=='__main__':
    goh=GUI()
    goh()






