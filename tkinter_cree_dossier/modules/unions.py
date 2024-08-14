from tkinter_cree_dossier.modules._etc import *

class UNION_3(Module_Mdl):
	bg, fg = 'light pink', 'black'
	nom = "[UNION 3]"
	X, Y = [0,0,0], [0]
	X_noms, Y_noms = ["X0", "X1", "X2"], ["Y"]
	params = {
	}
	def cree_ix(self):
		#	Params
		X0 = self.X[0]
		X1 = self.X[1]
		X2 = self.X[2]
		#
		Y = self.Y[0]

		#	------------------

		self.elements = {
			'u01' : MODULE_i_Union(X=[X0,X1], Y=[X0+X1], params={}).cree_ix(),
			'u23' : MODULE_i_Union(X=[X0+X1,X2], Y=[X0+X1+X2], params={}).cree_ix(),
		}

		self.connections = {
			'u01' : {0 : None, 1 : None},
			'u23' : {0 : ('u01',0), 1 : None},
		}

		self.cree_elements_connections()
		return self.ix

class UNION_4(Module_Mdl):
	bg, fg = 'light pink', 'black'
	nom = "[UNION 4]"
	X, Y = [0,0,0,0], [0]
	X_noms, Y_noms = ["X0", "X1", "X2", "X3"], ["Y"]
	params = {
	}
	def cree_ix(self):
		#	Params
		X0 = self.X[0]
		X1 = self.X[1]
		X2 = self.X[2]
		X3 = self.X[3]
		#
		Y = self.Y[0]

		#assert Y == (X0+X1+X2+X3)
		#assert X0==X1==X2==X3

		#	------------------

		self.elements = {
			'u01' : MODULE_i_Union(X=[X0,X1], Y=[X0+X1], params={}).cree_ix(),
			'u23' : MODULE_i_Union(X=[X2,X3], Y=[X2+X3], params={}).cree_ix(),
			'u0123' : MODULE_i_Union(X=[X0+X1,X2+X3], Y=[Y], params={}).cree_ix(),
		}

		self.connections = {
			'u01' : {0 : None, 1 : None},
			'u23' : {0 : None, 1 : None},
			'u0123' : {0 : ('u01',0), 1 : ('u23',0)},
		}

		self.cree_elements_connections()
		return self.ix
