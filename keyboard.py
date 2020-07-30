# add all button line wise 

# First Line Button

from tkinter import ttk
import tkinter as tk


FIRST_ROW = 5
ENTER = "Enter"

class Keyboard:

    def __init__(self, exp, equation, key, predictor, speech_to_text):
        self.exp = exp
        self.equation = equation
        self.equation.set(self.exp)
        self.key = key
        self.prev_hidden = None
        self.last_input = None
        self.predictor = predictor
        self.speech_to_text = speech_to_text
        self.top_k_words = [tk.StringVar(value='') for i in range(15)]
        self.top_k_sentences = [tk.StringVar(value='') for i in range(3)]

    def press(self, num):
        self.exp = self.exp + str(num)
        self.equation.set(self.exp)

    def press_reco(self, i):
        def press_reco_func():
            word = self.top_k_words[i].get()
            
            if word == ENTER:
                return self.action()

            self.exp = self.exp + word + " "
            self.equation.set(self.exp)
            self.update_words(word)
        return press_reco_func

    def press_reco_sentence(self, i):
        def press_reco_func():
            sentence = self.top_k_sentences[i].get()
            self.exp = sentence
            self.equation.set(self.exp)
        return press_reco_func

    def press_space(self):
        word = self.exp.split(" ")[-1].lower()
        self.exp = self.exp + " "
        self.equation.set(self.exp)
        self.update_words(word)

    def clear(self):
        self.exp = " "
        self.equation.set(self.exp)
        self.prev_hidden = None
        self.new_sentence(self.last_input)

    def action(self):
        self.speech_to_text.talk(self.exp)
        self.clear()

    def Tab(self):
        self.exp = " TAB : "
        self.equation.set(self.exp)

    def new_sentence(self, sentence):
        hidden = self.predictor.encode_sentence(sentence)
        self.prev_hidden = hidden
        self.update_words()
        
    def update_words(self, last_word=None):
        next_words, hidden = self.predictor.predict_next_word(self.prev_hidden, prev_word=last_word)
        for but, word in zip(self.top_k_words, next_words):
            if word == self.predictor.EOS:
                word = ENTER
            
            but.set(word)
        self.prev_hidden = hidden

    def set_up_reco(self):
        style = ttk.Style() 
        style.configure('Reco.TButton', font = 
                        ('calibri', 12, 'bold'), 
                        borderwidth = '4',
                        foreground = 'black') 
        for i, word in enumerate(self.top_k_words):
            row = i // 5 

            reco0 = ttk.Button(self.key, textvariable=word , width = 12, command = self.press_reco(i),
                                style='Reco.TButton')
            reco0.grid(row = FIRST_ROW - 3 + row , column = 2 * (i % 5), pady = 15, ipadx = 5 , ipady = 10, columnspan=2)

    def set_up_reco_sentences(self):
        style = ttk.Style() 
        style.configure('Reco.TButton', font = 
                        ('calibri', 12, 'bold'), 
                        borderwidth = '4',
                        foreground = 'black')
        for i, sentence in enumerate(self.top_k_sentences):
            reco0 = ttk.Button(self.key, textvariable=self.top_k_sentences[i] , width = 36,
                                command = self.press_reco_sentence(i), style='Reco.TButton')
            reco0.grid(row = FIRST_ROW - i - 1, column = 10, pady = 15, ipadx = 5 , ipady = 10, columnspan=4)
        
    def set_up_keyboard(self):
        q = ttk.Button(self.key,text = 'q' , width = 6, command = lambda : self.press('q'))
        q.grid(row = FIRST_ROW , column = 0, ipadx = 6 , ipady = 10)

        w = ttk.Button(self.key,text = 'w' , width = 6, command = lambda : self.press('w'))
        w.grid(row = FIRST_ROW , column = 1, ipadx = 6 , ipady = 10)

        E = ttk.Button(self.key,text = 'e' , width = 6, command = lambda : self.press('e'))
        E.grid(row = FIRST_ROW , column = 2, ipadx = 6 , ipady = 10)

        R = ttk.Button(self.key,text = 'r' , width = 6, command = lambda : self.press('r'))
        R.grid(row = FIRST_ROW , column = 3, ipadx = 6 , ipady = 10)

        T = ttk.Button(self.key,text = 't' , width = 6, command = lambda : self.press('t'))
        T.grid(row = FIRST_ROW , column = 4, ipadx = 6 , ipady = 10)

        Y = ttk.Button(self.key,text = 'y' , width = 6, command = lambda : self.press('y'))
        Y.grid(row = FIRST_ROW , column = 5, ipadx = 6 , ipady = 10)

        U = ttk.Button(self.key,text = 'u' , width = 6, command = lambda : self.press('u'))
        U.grid(row = FIRST_ROW , column = 6, ipadx = 6 , ipady = 10)

        I = ttk.Button(self.key,text = 'i' , width = 6, command = lambda : self.press('i'))
        I.grid(row = FIRST_ROW , column = 7, ipadx = 6 , ipady = 10)

        O = ttk.Button(self.key,text = 'o' , width = 6, command = lambda : self.press('o'))
        O.grid(row = FIRST_ROW , column = 8, ipadx = 6 , ipady = 10)

        P = ttk.Button(self.key,text = 'p' , width = 6, command = lambda : self.press('p'))
        P.grid(row = FIRST_ROW , column = 9, ipadx = 6 , ipady = 10)

        cur = ttk.Button(self.key,text = '{' , width = 6, command = lambda : self.press('{'))
        cur.grid(row = FIRST_ROW , column = 10 , ipadx = 6 , ipady = 10)

        cur_c = ttk.Button(self.key,text = '}' , width = 6, command = lambda : self.press('}'))
        cur_c.grid(row = FIRST_ROW , column = 11, ipadx = 6 , ipady = 10)

        back_slash = ttk.Button(self.key,text = '\\' , width = 6, command = lambda : self.press('\\'))
        back_slash.grid(row = FIRST_ROW , column = 12, ipadx = 6 , ipady = 10)


        clear = ttk.Button(self.key,text = 'Clear' , width = 6, command = self.clear)
        clear.grid(row = FIRST_ROW , column = 13, ipadx = 20 , ipady = 10)

        # Second Line Button



        A = ttk.Button(self.key,text = 'a' , width = 6, command = lambda : self.press('a'))
        A.grid(row = FIRST_ROW + 1 , column = 0, ipadx = 6 , ipady = 10)

        S = ttk.Button(self.key,text = 's' , width = 6, command = lambda : self.press('s'))
        S.grid(row = FIRST_ROW + 1 , column = 1, ipadx = 6 , ipady = 10)

        D = ttk.Button(self.key,text = 'd' , width = 6, command = lambda : self.press('d'))
        D.grid(row = FIRST_ROW + 1 , column = 2, ipadx = 6 , ipady = 10)

        F = ttk.Button(self.key,text = 'f' , width = 6, command = lambda : self.press('f'))
        F.grid(row = FIRST_ROW + 1 , column = 3, ipadx = 6 , ipady = 10)


        G = ttk.Button(self.key,text = 'g' , width = 6, command = lambda : self.press('g'))
        G.grid(row = FIRST_ROW + 1 , column = 4, ipadx = 6 , ipady = 10)


        H = ttk.Button(self.key,text = 'h' , width = 6, command = lambda : self.press('h'))
        H.grid(row = FIRST_ROW + 1 , column = 5, ipadx = 6 , ipady = 10)


        J = ttk.Button(self.key,text = 'j' , width = 6, command = lambda : self.press('j'))
        J.grid(row = FIRST_ROW + 1 , column = 6, ipadx = 6 , ipady = 10)


        K = ttk.Button(self.key,text = 'k' , width = 6, command = lambda : self.press('k'))
        K.grid(row = FIRST_ROW + 1 , column = 7, ipadx = 6 , ipady = 10)

        L = ttk.Button(self.key,text = 'l' , width = 6, command = lambda : self.press('l'))
        L.grid(row = FIRST_ROW + 1 , column = 8, ipadx = 6 , ipady = 10)


        semi_co = ttk.Button(self.key,text = ';' , width = 6, command = lambda : self.press(';'))
        semi_co.grid(row = FIRST_ROW + 1 , column = 9, ipadx = 6 , ipady = 10)


        d_colon = ttk.Button(self.key,text = '"' , width = 6, command = lambda : self.press('"'))
        d_colon.grid(row = FIRST_ROW + 1 , column = 10, ipadx = 6 , ipady = 10)


        enter = ttk.Button(self.key,text = 'Enter' , width = 6, command = self.action)
        enter.grid(row = FIRST_ROW + 1 , columnspan = 75, ipadx = 85 , ipady = 10)

        # third line Button

        Z = ttk.Button(self.key,text = 'z' , width = 6, command = lambda : self.press('z'))
        Z.grid(row = FIRST_ROW + 2 , column = 0, ipadx = 6 , ipady = 10)


        X = ttk.Button(self.key,text = 'x' , width = 6, command = lambda : self.press('x'))
        X.grid(row = FIRST_ROW + 2 , column = 1, ipadx = 6 , ipady = 10)


        C = ttk.Button(self.key,text = 'c' , width = 6, command = lambda : self.press('c'))
        C.grid(row = FIRST_ROW + 2 , column = 2, ipadx = 6 , ipady = 10)


        V = ttk.Button(self.key,text = 'v' , width = 6, command = lambda : self.press('v'))
        V.grid(row = FIRST_ROW + 2 , column = 3, ipadx = 6 , ipady = 10)

        B = ttk.Button(self.key, text= 'b' , width = 6 , command = lambda : self.press('b'))
        B.grid(row = FIRST_ROW + 2 , column = 4 , ipadx = 6 ,ipady = 10)


        N = ttk.Button(self.key,text = 'n' , width = 6, command = lambda : self.press('n'))
        N.grid(row = FIRST_ROW + 2 , column = 5, ipadx = 6 , ipady = 10)


        M = ttk.Button(self.key,text = 'm' , width = 6, command = lambda : self.press('m'))
        M.grid(row = FIRST_ROW + 2 , column = 6, ipadx = 6 , ipady = 10)


        left = ttk.Button(self.key,text = '<' , width = 6, command = lambda : self.press('<'))
        left.grid(row = FIRST_ROW + 2 , column = 7, ipadx = 6 , ipady = 10)


        right = ttk.Button(self.key,text = '>' , width = 6, command = lambda : self.press('>'))
        right.grid(row = FIRST_ROW + 2 , column = 8, ipadx = 6 , ipady = 10)


        slas = ttk.Button(self.key,text = '/' , width = 6, command = lambda : self.press('/'))
        slas.grid(row = FIRST_ROW + 2 , column = 9, ipadx = 6 , ipady = 10)


        q_mark = ttk.Button(self.key,text = '?' , width = 6, command = lambda : self.press('?'))
        q_mark.grid(row = FIRST_ROW + 2 , column = 10, ipadx = 6 , ipady = 10)


        coma = ttk.Button(self.key,text = ',' , width = 6, command = lambda : self.press(','))
        coma.grid(row = FIRST_ROW + 2 , column = 11, ipadx = 6 , ipady = 10)

        dot = ttk.Button(self.key,text = '.' , width = 6, command = lambda : self.press('.'))
        dot.grid(row = FIRST_ROW + 2 , column = 12, ipadx = 6 , ipady = 10)

        shift = ttk.Button(self.key,text = 'Shift' , width = 6, command = lambda : self.press('Shift'))
        shift.grid(row = FIRST_ROW + 2 , column = 13, ipadx = 20 , ipady = 10)

        #Fourth Line Button


        ctrl = ttk.Button(self.key,text = 'Ctrl' , width = 6, command = lambda : self.press('Ctrl'))
        ctrl.grid(row = FIRST_ROW + 3 , column = 0, ipadx = 6 , ipady = 10)


        Fn = ttk.Button(self.key,text = 'Fn' , width = 6, command = lambda : self.press('Fn'))
        Fn.grid(row = FIRST_ROW + 3 , column = 1, ipadx = 6 , ipady = 10)


        window = ttk.Button(self.key,text = 'Window' , width = 6, command = lambda : self.press('Window'))
        window.grid(row = FIRST_ROW + 3 , column = 2 , ipadx = 6 , ipady = 10)

        Alt = ttk.Button(self.key,text = 'Alt' , width = 6, command = lambda : self.press('Alt'))
        Alt.grid(row = FIRST_ROW + 3 , column = 3 , ipadx = 6 , ipady = 10)

        space = ttk.Button(self.key,text = 'Space' , width = 6, command = lambda : self.press_space())
        space.grid(row = FIRST_ROW + 3 , columnspan = 14 , ipadx = 160 , ipady = 10)

        Alt_gr = ttk.Button(self.key,text = 'Alt Gr' , width = 6, command = lambda : self.press('Alt Gr'))
        Alt_gr.grid(row = FIRST_ROW + 3 , column = 10 , ipadx = 6 , ipady = 10)

        open_b = ttk.Button(self.key,text = '(' , width = 6, command = lambda : self.press('('))
        open_b.grid(row = FIRST_ROW + 3 , column = 11 , ipadx = 6 , ipady = 10)

        close_b = ttk.Button(self.key,text = ')' , width = 6, command = lambda : self.press(')'))
        close_b.grid(row = FIRST_ROW + 3 , column = 12 , ipadx = 6 , ipady = 10)


        tap = ttk.Button(self.key,text = 'Tab' , width = 6, command = self.Tab)
        tap.grid(row = FIRST_ROW + 3 , column = 13 , ipadx = 20 , ipady = 10)

        self.set_up_reco()
        self.set_up_reco_sentences()
