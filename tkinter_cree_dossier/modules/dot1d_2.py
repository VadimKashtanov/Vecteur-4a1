from tkinter_cree_dossier.modules._etc import *

class DOT1D_2X(Module_Mdl):
	bg, fg = 'light yellow', 'black'
	nom = "[DOT1D_2X:f(ax0+bx1+c)]"
	X, Y = [0,0], [0]
	X_noms, Y_noms = ["X0", "X1"], ["Y"]
	params = {
		'activ' : 0,
		'C0' : 1,
	}
	def cree_ix(self):
		#	Params
		X0 = self.X[0]
		X1 = self.X[1]
		Y  = self.Y[0]

		activ = self.params['activ']
		C0 = self.params['C0']

		assert X0 % C0 == 0
		assert X1 % C0 == 0
		assert Y  % C0 == 0

		#	------------------

		self.elements = {
			'a'    : MODULE_i_MatMul_Poid_AP (X=[X0],  Y=[Y], params={'Ax':int(X0/C0), 'Ay':1, 'Bx':int(Y/C0), 'C0':C0}, do=0,dc=self.dc).cree_ix(),
			'b'    : MODULE_i_MatMul_Poid_AP (X=[X1],  Y=[Y], params={'Ax':int(X1/C0), 'Ay':1, 'Bx':int(Y/C0), 'C0':C0}, do=0,dc=self.dc).cree_ix(),
			's'    : MODULE_i_Somme          (X=[Y,Y], Y=[Y], params={}, do=self.do,dc=0).cree_ix(),
			'f(s)' : MODULE_i_Activation_Poid(X=[Y],   Y=[Y], params={'activ':activ}, do=self.do,dc=self.dc).cree_ix(),
		}

		#	======================

		self.connections = {
			'a' : {0 : None},
			'b' : {0 : None},
			's' : {
				0 : ('a', 0),
				1 : ('b', 0)
			},
			'f(s)' : {
				0 : ('s', 0)
			},
		}

		self.cree_elements_connections()
		return self.ix