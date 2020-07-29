import tkinter as tk
from keyboard import Keyboard
from tkinter import ttk

from pred import *
from speechtotext import SpeechToText


key = tk.Tk()              # key window name

key.title('Alis Keyboard')
key.iconbitmap('alis.bmp') 

key.configure(bg="#2C3E50")


# Size window size
key.geometry('1010x250')         # normal size
key.maxsize(width=1010, height=500)      # maximum size
key.minsize(width= 1010 , height = 500)     # minimum size
# end window size


# entry box
equation = tk.StringVar()
Dis_entry = ttk.Entry(key,state= 'readonly',textvariable = equation)
Dis_entry.grid(row=1, rowspan= 1 , columnspan = 100, ipadx = 999 , ipady = 20)
# end entry box


predictor = NextWordPredictor('models/encoder.pkl', 'models/decoder.pkl',
                              'models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin')
# predictor = None

# Speech to text section
speech_to_text = SpeechToText()


keyboard = Keyboard("", equation, key, predictor, speech_to_text)
keyboard.set_up_keyboard()


def input_callback():
    value = speech_to_text.recognize()
    speech_var.set(value)
    keyboard.last_input = value
    keyboard.new_sentence(value)

speech_var = tk.StringVar()
speech_entry = ttk.Entry(key,state= 'readonly',textvariable = speech_var)
speech_entry.grid(row=0, rowspan=1, columnspan = 2, column=2)
tk.Button(key, 
          text='Listen to someone', 
          command=input_callback).grid(row=0, column=0, sticky=tk.W, pady=4, columnspan=2)


# tk.Label(key, text="Input").grid(row=0, ipady=10)
# e1 = tk.Entry(key)
# e1.grid(row=0, rowspan= 1, column=1, columnspan=5)

# def input_callback():
#     value = e1.get()
#     keyboard.last_input = value
#     keyboard.new_sentence(value)

# tk.Button(key, 
#           text='Select', 
#           command=input_callback).grid(row=0, column=2, sticky=tk.W, pady=4)



key.mainloop()  # using ending point