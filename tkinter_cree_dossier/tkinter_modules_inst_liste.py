from tkinter_cree_dossier.tkinter_mdl import Module_Mdl
from tkinter_cree_dossier.tkinter_dico_inst import Dico
#
from tkinter_cree_dossier.tkinter_insts import *

conn = lambda sortie,inst,entree: (sortie, (inst,entree))

modules_inst = []

for i in liste_insts+[i_Y]:
	nom_classe = i.__name__
	s = f"""
class MODULE_{nom_classe}(Module_Mdl):	#	A+B
	nom = "i:{i.nom}"
	X, Y = {list(i.X)}, [0]
	X_noms, Y_noms = {["X" for _ in i.X]}, ["Y"]
	params = {{
	"""
	for p in i.params_str:
		s += f"""
		'{p}' : 0,"""
	s += f"""
	}}
	def cree_ix(self):
		#	Params
		X = self.X
		Y = self.Y[0]
		params = [p for _,p in self.params.items()]

		#	------------------

		self.ix = [
			Dico(i={nom_classe}, X=X, x={[None for _ in i.X]}, xt={[None for _ in i.X]}, y=Y, p=params, sortie=True)
		]

		return self.ix

modules_inst += [MODULE_{nom_classe}]
"""
	exec(s)