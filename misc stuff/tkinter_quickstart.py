import tkinter as tk
import pprint

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(expand=1) # position-management mechanism
        self.create_widgets()

    def create_widgets(self):
        self.button = tk.Button(self)
        # config method to update multiple attributes at once
        self.button.config(text="Hello World\n(click me!)", command=self.say_hi)
        # print options supported by the widget
        print(pprint.pformat(self.button.config()))
        # self.button["text"] = "Hello World\n(click me!)"
        # self.button["command"] = self.say_hi
        self.button.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("Hello Everyone!")

root = tk.Tk()  # instantiate Tk class once in an application
app = Application(master=root)
app.mainloop()