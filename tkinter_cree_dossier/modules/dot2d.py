from tkinter_cree_dossier.modules._etc import *

class DOT2D_AP(Module_Mdl):
	bg, fg = 'light grey', 'black'
	nom = "[DOT2D A@P:f(ax+b)]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'Ax' : 1,
		'Ay' : 1,
		'Bx' : 1,
		'activ' : 0,
		'C0' : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		Ax = self.params['Ax']
		Ay = self.params['Ay']
		Bx = self.params['Bx']
		activ = self.params['activ']
		C0 = self.params['C0']

		assert X % C0 == 0
		assert Y % C0 == 0

		assert X == (Ax*Ay*C0)
		assert Y == (Ay*Bx*C0)

		#	------------------

		self.elements = {
			's'    : MODULE_i_MatMul_Poid_AP (X=[X], Y=[Y], params={'Ax':Ax, 'Ay':Ay, 'Bx':Bx, 'C0':C0}).cree_ix(),
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


class DOT2D_PA(Module_Mdl):
	bg, fg = 'light grey', 'black'
	nom = "[DOT2D P@A:f(xa+b)]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'Ax' : 1,
		'Ay' : 1,
		'Bx' : 1,
		'activ' : 0,
		'C0' : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		Ax = self.params['Ax']
		Ay = self.params['Ay']
		Bx = self.params['Bx']
		activ = self.params['activ']
		C0 = self.params['C0']

		assert X % C0 == 0
		assert Y % C0 == 0

		assert X == (Bx*Ax*C0)
		assert Y == (Ay*Bx*C0)

		#	------------------

		self.elements = {
			's'    : MODULE_i_MatMul_Poid_PA (X=[X], Y=[Y], params={'Ax':Ax, 'Ay':Ay, 'Bx':Bx, 'C0':C0}, do=0,dc=self.dc).cree_ix(),
			'f(s)' : MODULE_i_Activation_Poid(X=[Y], Y=[Y], params={'activ':activ}, do=self.do,dc=self.dc).cree_ix(),
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