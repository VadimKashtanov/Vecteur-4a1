#! /usr/bin/python3

import tkinter as tk
import struct  as st
from tkinter import messagebox
from tkinter import filedialog

from tkinter_cree_dossier.tkinter_modules_inst_liste import modules_inst # Modules : Instructions Induviduelle
from tkinter_cree_dossier.tkinter_modules_liste      import modules      # Modules : Assemblage d'instructions

from tkinter_cree_dossier.tkinter_insts              import liste_insts

modules_models = modules_inst + modules

##############################################################

def rgb(r,g,b):
	return '#%02x%02x%02x' % (r,g,b)

##############################################################

class Entree(tk.Frame):
	def __init__(self, parent, A, val_ini, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.A = A
		self.l = tk.Label(self, text=A, bg='white', font=("DejaVu Sans Mono", 12))
		self.val = tk.StringVar()
		self.val.set(val_ini)
		self.e = tk.Entry(self, textvariable=self.val, width=8)
		#
		self.l.grid(row=0,column=0, sticky='nsew')
		self.e.grid(row=0,column=1, sticky='nsew')