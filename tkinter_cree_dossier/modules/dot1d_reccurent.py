from tkinter_cree_dossier.modules._etc import *

from tkinter_cree_dossier.modules.dot1d import *

class DOT1D_RECCURENT(Module_Mdl):
	bg, fg = 'light yellow', 'black'
	nom = "[DOT1D RECCURENT]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X0"], ["Y"]
	params = {
		'activ' : 0,
		'C0' : 1,
	}
	def cree_ix(self):
		#	Params
		X0 = self.X[0]
		Y  = self.Y[0]

		activ = self.params['activ']
		C0 = self.params['C0']

		assert X0 % C0 == 0
		assert Y  % C0 == 0

		#	------------------

		self.elements = {
			'a'    : MODULE_i_MatMul_Poid_AP (X=[X0],  Y=[Y], params={'Ax':int(X0/C0), 'Ay':1, 'Bx':int(Y/C0), 'C0':C0}, do=0,dc=self.dc).cree_ix(),
			'b'    : MODULE_i_MatMul_Poid_AP (X=[Y ],  Y=[Y], params={'Ax':int(Y /C0), 'Ay':1, 'Bx':int(Y/C0), 'C0':C0}, do=0,dc=self.dc).cree_ix(),
			's'    : MODULE_i_Somme          (X=[Y,Y], Y=[Y], params={}, do=self.do,dc=0).cree_ix(),
			'f(s)' : MODULE_i_Activation_Poid(X=[Y],   Y=[Y], params={'activ':activ}, do=self.do,dc=self.dc).cree_ix(),
		}

		#	======================

		self.connections = {
			'a' : {0 : None},
			'b' : {0 : ('f(s)', -1)},
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

class DOT1D_RECCURENT__CHAINE(CHAINE):
	img = img_chaine
	bg, fg = 'light yellow', 'black'
	nom = "[DOT1D RECCURENT] CHAINE"
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
	ELEMENT = DOT1D_RECCURENT

#####################################################################

class DOT1D_RECCURENT_N(Module_Mdl):
	bg, fg = 'light yellow', 'black'
	nom = "[DOT1D RECCURENT N]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X0"], ["Y"]
	params = {
		'N-analyse' : 0,
		'activ' : 0,
		'C0' : 1,
	}
	def cree_ix(self):
		#	Params
		X0 = self.X[0]
		Y  = self.Y[0]

		N_analyse = self.params['N-analyse']
		activ = self.params['activ']
		C0 = self.params['C0']

		assert X0 % C0 == 0
		assert Y  % C0 == 0

		#	------------------

		self.elements = {
			'a'    : DOT1D__CHAINE (X=[X0],  Y=[Y], params={'H':Y, 'N':N_analyse, 'C0':C0, 'activ':activ}, do=0,dc=self.dc).cree_ix(),
			'b'    : DOT1D__CHAINE (X=[Y ],  Y=[Y], params={'H':Y, 'N':N_analyse, 'C0':C0, 'activ':activ}, do=0,dc=self.dc).cree_ix(),
			's'    : MODULE_i_Somme          (X=[Y,Y], Y=[Y], params={}, do=self.do,dc=0).cree_ix(),
			'f(s)' : MODULE_i_Activation_Poid(X=[Y],   Y=[Y], params={'activ':activ}, do=self.do,dc=self.dc).cree_ix(),
		}

		#	======================

		self.connections = {
			'a' : {0 : None},
			'b' : {0 : ('f(s)', -1)},
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

class DOT1D_RECCURENT_N__CHAINE(CHAINE):
	img = img_chaine
	bg, fg = 'light yellow', 'black'
	nom = "[DOT1D RECCURENT N] CHAINE"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'N-analyse' : 1,
		'H' : 0,
		'N' : 1,
		'C0' : 1,
		'activ' : 0
	}
	#	--------------------------
	H       = 'H'
	N       = 'N'
	ELEMENT = DOT1D_RECCURENT_N