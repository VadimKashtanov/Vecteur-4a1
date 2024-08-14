#! /usr/bin/python3

from tkinter_cree_dossier.application._etc import *

from tkinter_cree_dossier.application.boutons    import Bouttons
from tkinter_cree_dossier.application.linecanvas import LineCanvas
from tkinter_cree_dossier.application.modules    import Modules

from tkinter_cree_dossier.application.linecanvas import DraggableFrame

class Application(tk.Tk):
	
	frames = []

	vraie_sortie = 0

	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.geometry("1800x1120")

		self.canvas = LineCanvas(
			self, self,
			width=1465, height=840,
			scrollregion=(0, 0, 1540, 500),
			bg=rgb(240,240,240),
		)
		self.boutons = Bouttons(
			self, self,
		)
		self.modules = Modules(
			self, self,
		)

		self.canvas  .grid(row=0, column=0, sticky='nsew')
		self.boutons .grid(row=0, column=1, sticky='nsew', rowspan=2)
		self.modules .grid(row=1, column=0, sticky='nsew')

		#	---- Rapide ----
		self.bind('+', self.ajouter_une_connection)
		self.bind('a', self.ajouter_une_connection)

		def change_focus(event):
			if (not type(event.widget) in [str]):
				event.widget.focus_set()

		self.bind_all('<Button>', change_focus)

	def supprimer_toutes_les_connections(self, *k):
		for i in range(len(self.canvas.connections)):
			del self.canvas.connections[0]

		self.canvas.update_lines()

	def suppr_une_frame(self, numero):
		f = self.trouver_frame(numero)
		f.pack_forget()
		f.destroy()
		del self.frames[self.frames.index(f)]
		#
		for c in range(len(self.canvas.connections)):
			(iA,sA), (iB, eB), t = self.canvas.connections[c]
			if iB==numero or iA==numero:
				self.canvas.connections[c] = None
		#
		self.canvas.connections = [c for c in self.canvas.connections if c != None]
		#
		self.canvas.update()
		self.canvas.update_lines()

	def update_frames(self):
		for f in self.frames:
			f.mettre_a_jour_module()

	def changer_ordre(self):
		de = eval(self.ordre_de.val.get())
		a  = eval(self.ordre_a.val.get())

		self.canvas.connections[de], self.canvas.connections[a] = self.canvas.connections[a], self.canvas.connections[de]

		self.canvas.update()
		self.canvas.update_lines()

	def ajouter_une_connection(self, *k):
		iA, sA = eval(self.boutons.connections.instA.get()), eval(self.boutons.connections.sortieA.get())
		iB, eB = eval(self.boutons.connections.instB.get()), eval(self.boutons.connections.entréeB.get())
		t = int(self.boutons.connections.t_A.get())
		self.canvas.ajouter_connections((iA,sA), (iB, eB), t)

	def supprimer_une_connection(self):
		iA, sA = eval(self.boutons.connections.instA.get()), eval(self.boutons.connections.sortieA.get())
		iB, eB = eval(self.boutons.connections.instB.get()), eval(self.boutons.connections.entréeB.get())
		t = eval(self.boutons.connections.t_A.get())
		if [(iA,sA), (iB,eB), t] in self.canvas.connections:
			del self.canvas.connections[self.canvas.connections.index([(iA,sA), (iB,eB),t])]
			self.canvas.update_lines()
		else:
			messagebox.showwarning('Attention', f"Il n'existe pas de connection iA={iA} sA={sA} iB={iB} eB={eB}")

	def trouver_frame(self, numero):
		for f in self.frames:
			if f.numero == numero:
				return f
		raise Exception(f"Pas trouvé le numéro {numero}")

	def numeros(self):
		return [f.numero for f in self.frames]

	def add_frame(self, module, x=0, y=0):
		frame = DraggableFrame(
			self, self, x, y, module, self.prochain_numero_a_donner, 
			width=100, height=135, bg=rgb(255,255,255))
		frame.pack_propagate(0)
		self.frames.append(frame)
		self.canvas.update_lines()
		#
		self.prochain_numero_a_donner += 1

	def mise_a_jour_sortie(self):
		self.vraie_sortie = eval(self.boutons.sortie_exacte.la_sortie.val.get())
		self.re_ordonner_frames()

	#	========================================================================

	def move_objects_up(self):
		self.y -= 1
		for frame in self.frames:
			frame.place_configure(y=frame.winfo_y() - -400)
		self.canvas.update()
		self.canvas.update_lines()

	def move_objects_down(self):
		self.y += 1
		for frame in self.frames:
			frame.place_configure(y=frame.winfo_y() + -400)
		self.canvas.update()
		self.canvas.update_lines()

	def move_objects_left(self):
		self.x -= 1
		for frame in self.frames:
			frame.place_configure(x=frame.winfo_x() - -400)
		self.canvas.update()
		self.canvas.update_lines()

	def move_objects_right(self):
		self.x += 1
		for frame in self.frames:
			frame.place_configure(x=frame.winfo_x() + -400)
		self.canvas.update()
		self.canvas.update_lines()

	def re_ordonner_frames(self):
		self.update_frames()
		#
		frames_ordonnées = []
		for (iA,sA),(iB,eB),t in self.canvas.connections:
			if not self.trouver_frame(iA) in frames_ordonnées:
				frames_ordonnées += [self.trouver_frame(iA)]
		for (iA,sA),(iB,eB),t in self.canvas.connections:
			if not self.trouver_frame(iB) in frames_ordonnées:
				frames_ordonnées += [self.trouver_frame(iB)]
		if not self.trouver_frame(self.vraie_sortie) in frames_ordonnées:
			frames_ordonnées += [self.trouver_frame(self.vraie_sortie)]

		a_supprimer = []
		for f in self.frames:
			if not f in frames_ordonnées:
				a_supprimer += [f]
		for f in a_supprimer:
			f.suppr_la_frame()

		nouveaux = {
			self.frames[i].numero : frames_ordonnées.index(self.frames[i]) for i in range(len(self.frames))
		}

		for i,(ancien,nouveau) in enumerate(nouveaux.items()):
			self.frames[i].numero = nouveau

		for c in range(len(self.canvas.connections)):
			(iA,sA), (iB,eB), t = self.canvas.connections[c]
			for i,(ancien,nouveau) in enumerate(nouveaux.items()):
				if iA == ancien:
					iA = nouveau
					break

			for i,(ancien,nouveau) in enumerate(nouveaux.items()):
				if iB == ancien:
					iB = nouveau
					break

			self.canvas.connections[c] = [(iA,sA), (iB,eB), t]

		#	Re-ordonner la liste self.frames
		self.frames = [self.trouver_frame(i) for i in range(len(self.frames))]
		self.prochain_numero_a_donner = len(self.frames)

		#	--
		self.canvas.update()
		self.update_frames()
		self.canvas.update_lines()

if __name__ == "__main__":
	Application().mainloop()