from tkinter_cree_dossier.application._etc import *

class Modules(tk.Frame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, *args, **kwargs)
		self.application = application
		self.images = []
		for i_m in range(len(modules)):
			m = modules[i_m]
			#
			if m.img == None:
				self.images += [None]
			else:
				self.images += [tk.PhotoImage(file=m.img)]
			#
			tk.Button(
				self,
				text=f'{m.nom}',
				bg=m.bg,
				fg=m.fg,
				image=self.images[-1],
				compound="left",
				command=lambda _m=m:self.application.add_frame(_m())
			).grid(
				row=i_m%9,
				column=i_m//9,
				sticky='nsew'
			)