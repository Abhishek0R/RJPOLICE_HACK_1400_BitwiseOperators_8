from tkinter import *
import imageio
import tensorflow
import numpy as np
from PIL import Image
import cv2

BG_GRAY = "#C9AE5D"
BG_COLOR = "#F7E7CE"
TEXT_COLOR = "#665D1E"

FONT = "Georgia 18"
FONT_BOLD = "Georgia 18 bold"


class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("MAIN")
        self.window.resizable(width=True, height=True)
        self.window.configure(width=1920, height=1080, bg=BG_COLOR)


        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Deep Fake Detection", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)


        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)


        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow")

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        bottom_label = Label(self.window, bg=BG_GRAY, height=100)
        bottom_label.place(relwidth=1, rely=0.825)
        self.msg_entry = Entry(bottom_label, bg="#BAB86C", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        send_button = Button(bottom_label, text="Submit", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

        m="You"
        L="HELLO"
        K="Bot"
        T="PREDICTION OCCURS HERE"
        self.msg_entry.delete(0, END)
        msg1 = f"{m}: {L}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{K}: {T}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()


def run(source=None):
    model = tensorflow.keras.models.load_model('MESO4.h5')
    image = imageio.imread(source)




if __name__ == "__main__":
    #run(source='101a.jpg')
    app = ChatApplication()
    app.run()

