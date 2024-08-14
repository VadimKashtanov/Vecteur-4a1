from tkinter_cree_dossier.modules._etc import *

class SPLIT_4(Module_Mdl):
	bg, fg = 'light pink', 'black'
	nom = "[Split 4]"
	X, Y = [0], [0,0,0,0]
	X_noms, Y_noms = ["X"], ["Y0", "Y1", "Y2", "Y3"]
	params = {
	}
	def cree_ix(self):
		#	Params
		Y0 = self.Y[0]
		Y1 = self.Y[1]
		Y2 = self.Y[2]
		Y3 = self.Y[3]
		#
		X = self.X[0]

		assert X == (Y0+Y1+Y2+Y3)
		assert Y0==Y1==Y2==Y3

		#	------------------

		self.elements = {
			'x' : MODULE_i_Y(X=[X], Y=[X], params={}).cree_ix(),
			#
			's0' : MODULE_i_Select_Vect(X=[X], Y=[Y0], params={'Vect':Y0, 'N':0}).cree_ix(),
			's1' : MODULE_i_Select_Vect(X=[X], Y=[Y1], params={'Vect':Y1, 'N':1}).cree_ix(),
			's2' : MODULE_i_Select_Vect(X=[X], Y=[Y2], params={'Vect':Y2, 'N':2}).cree_ix(),
			's3' : MODULE_i_Select_Vect(X=[X], Y=[Y3], params={'Vect':Y3, 'N':3}).cree_ix(),
		}

		self.connections = {
			'x' : {0 : None},
			's0' : {0 : ('x',0)},
			's1' : {0 : ('x',0)},
			's2' : {0 : ('x',0)},
			's3' : {0 : ('x',0)},
		}

		self.sorties = [-4,-3,-2,-1]

		self.cree_elements_connections()
		return self.ix
