import tkinter as tk
class GUI():
    def __init__(self):
        self.root=tk.Tk()
        self.root.geometry("1000x1000")
        self.root.configure(bg="black")
        self.title = tk.Label(text="Kenneth Johnson's Port Reader",foreground="white", background="black")
        self.title.config(font=("Arial", 20))
        self.title.pack()
        self.root.mainloop()

if __name__=='__main__':
    goh=GUI()

# from tkinter import filedialog
# from tkinter import *

# def browse_button():
#     # Allow user to select a directory and store it in global var
#     # called folder_path
#     filename = filedialog.askdirectory()
#     folder_path.set(filename)
#     print(filename)


# root = Tk()
# folder_path = StringVar()
# lbl1 = Label(master=root,textvariable=folder_path)
# lbl1.grid(row=0, column=1)
# button2 = Button(text="Browse", command=browse_button)
# button2.grid(row=0, column=3)




