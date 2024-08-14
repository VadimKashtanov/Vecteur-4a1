from tkinter_cree_dossier.application._etc import *

class Instructions(tk.LabelFrame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, text='Instructions', *args, **kwargs)
		self.application = application

		for i_m in range(len(modules_inst)): #modules_models:
			m = modules_inst[i_m]
			tk.Button(
				self,
				text=f'{m.nom}',
				command=lambda _m=m:self.application.add_frame(_m())
			).grid(
				row=i_m//2,
				column=i_m%2,
				sticky='nsew'
			)

class Deplacement(tk.LabelFrame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, text='Deplacement', *args, **kwargs)
		self.application = application

		# Load arrow images
		up_img    = tk.PhotoImage(file="tkinter_cree_dossier/img/arrow_up.png"   )
		down_img  = tk.PhotoImage(file="tkinter_cree_dossier/img/arrow_down.png" )
		left_img  = tk.PhotoImage(file="tkinter_cree_dossier/img/arrow_left.png" )
		right_img = tk.PhotoImage(file="tkinter_cree_dossier/img/arrow_right.png")

		# Create arrow buttons
		self.application.x, self.application.y = 0, 0
		move_up_btn    = tk.Button(self, image=up_img,    command=self.application.move_objects_up   )
		move_up_btn.grid   (row=0, column=1)
		move_down_btn  = tk.Button(self, image=down_img,  command=self.application.move_objects_down )
		move_down_btn.grid (row=1, column=1)
		move_left_btn  = tk.Button(self, image=left_img,  command=self.application.move_objects_left )
		move_left_btn.grid (row=1, column=0)
		move_right_btn = tk.Button(self, image=right_img, command=self.application.move_objects_right)
		move_right_btn.grid(row=1, column=2)

		# Keep reference to the images to prevent garbage collection
		self.arrow_images = [up_img, down_img, left_img, right_img]

		# Keep references to the buttons
		self.arrow_buttons = [move_up_btn, move_down_btn, move_left_btn, move_right_btn]

class Connections(tk.LabelFrame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, text='Connections', *args, **kwargs)
		self.application = application
		self.parent = parent

		tk.Label(self, text='Inst ').grid(row=0, column=1)
		tk.Label(self, text='Point').grid(row=0, column=2)
		tk.Label(self, text='A'    ).grid(row=1, column=0)
		tk.Label(self, text='B'    ).grid(row=2, column=0)

		tk.Label(self, text='[t]').grid(row=0, column=3)

		self.instA = tk.StringVar(); self.sortieA = tk.StringVar();
		self.instB = tk.StringVar(); self.entréeB = tk.StringVar();
		self.instA.set('0');         self.sortieA.set('0');
		self.instB.set('0');         self.entréeB.set('0');
		self.e_instA   = tk.Entry(self, textvariable=self.instA,   width=8)
		self.e_instB   = tk.Entry(self, textvariable=self.instB,   width=8)
		self.e_sortieA = tk.Entry(self, textvariable=self.sortieA, width=8)
		self.e_entréeB = tk.Entry(self, textvariable=self.entréeB, width=8)
		self.e_instA.grid  (row=1,column=1)
		self.e_instB.grid  (row=2,column=1)
		self.e_sortieA.grid(row=1,column=2)
		self.e_entréeB.grid(row=2,column=2)

		self.t_A   = tk.StringVar(); self.t_A.set('0')
		self.e_t_A = tk.Entry(self, textvariable=self.t_A,   width=8) 
		self.e_t_A.grid(row=1, column=3)

		tk.Button(self, text="+", fg=rgb(0,128,0), command=self.application.ajouter_une_connection  ).grid(row=3, column=1)
		tk.Button(self, text="x", fg=rgb(255,0,0), command=self.application.supprimer_une_connection).grid(row=3, column=3)

class Suppr_Conns(tk.LabelFrame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, text='Supprimer les connections', *args, **kwargs)
		self.application = application

		tk.Button(
			self,
			text='Supprimer les connections',
			fg='red',
			command=self.application.supprimer_toutes_les_connections
		).pack(fill=tk.X)

"""class Sortie_Exacte(tk.LabelFrame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, text='La Sortie Model', *args, **kwargs)
		self.application = application

		self.vraie_sortie = 0
		self.la_sortie = Entree(self, 'Sortie du Modele : ', '0')
		self.la_sortie.grid(row=0, column=0, sticky='nsew')
		tk.Button(
			self,
			text='Appliquer',
			command=self.application.mise_a_jour_sortie
		).grid(row=0, column=1, sticky='nsew')"""

from tkinter_cree_dossier.application.fichiers   import Fichiers

class Bouttons(tk.Frame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, *args, **kwargs)
		self.application = application

		self.instructions  = Instructions (self, self.application)
		self.deplacement   = Deplacement  (self, self.application)
		self.connections   = Connections  (self, self.application)
		self.suppr_tout    = Suppr_Conns  (self, self.application)
		#
		#self.sortie_exacte = Sortie_Exacte(self, self.application)
		#
		self.fichiers      = Fichiers     (self, self.application)

		self.instructions .pack(fill=tk.X)
		self.deplacement  .pack(fill=tk.X)
		self.connections  .pack(fill=tk.X)
		self.suppr_tout   .pack(fill=tk.X)
		#self.sortie_exacte.pack(fill=tk.X)
		self.fichiers     .pack(fill=tk.X)