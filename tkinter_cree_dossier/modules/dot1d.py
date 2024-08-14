from tkinter_cree_dossier.modules._etc import *

class DOT1D(Module_Mdl):
	bg, fg = 'white', 'black'
	nom = "[DOT1D:f(ax+b)]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'activ' : 0,
		'C0' : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		activ = self.params['activ']
		C0 = self.params['C0']

		assert X % C0 == 0
		assert Y % C0 == 0

		#	------------------

		self.elements = {
			's'    : MODULE_i_MatMul_Poid_AP (X=[X], Y=[Y], params={'Ax':int(X/C0), 'Ay':1, 'Bx':int(Y/C0), 'C0':C0}).cree_ix(),
			'f(s)' : MODULE_i_Activation_Poid(X=[Y], Y=[Y], params={'activ':activ}).cree_ix(),
		}

		self.connections = {
			's' : {
				0 : None,
			},
			'f(s)' : {
				0 : ('s', 0)
			},
		}

		self.cree_elements_connections()
		return self.ix

class DOT1D__CHAINE(CHAINE):
	img = img_chaine
	bg, fg = 'white', 'black'
	nom = "[DOT1D:f(ax+b)] CHAINE"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'H' : 0,
		'N' : 1,
		'C0' : 1,
		'activ' : 0
	}
	#	--------------------------
	H       = 'H'
	N       = 'N'
	ELEMENT = DOT1D
