#c = ~c + ~g
#h = ~tanh(c)
#avec ... = x@P + c@P + P

#c = ~c + ~g
#h = ~tanh(c)

from tkinter_cree_dossier.modules._etc import *

from tkinter_cree_dossier.modules.dot1d import *
from tkinter_cree_dossier.modules.dot1d_2 import *

class LSTM1D_PEEPHOLE(Module_Mdl):
	bg, fg = 'light yellow', 'black'
	nom = "[LSTM_PEEPHOLE]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'C0'    : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		C0 = self.params['C0']

		assert X % C0 == 0
		assert Y % C0 == 0

		#	------------------

		self.elements = {
			'x' : MODULE_i_Y(X=[X], Y=[X], params={}, do=0, dc=0).cree_ix(),
			#
			'f' : DOT1D_2X(X=[X,Y], Y=[Y], params={'activ':1, 'C0':C0}, do=self.do,dc=self.dc).cree_ix(),
			'i' : DOT1D_2X(X=[X,Y], Y=[Y], params={'activ':1, 'C0':C0}, do=self.do,dc=self.dc).cree_ix(),
			'u' : DOT1D   (X=[X],   Y=[Y], params={'activ':0, 'C0':C0}, do=self.do,dc=self.dc).cree_ix(),
			#
			'c' : AB_plus_CD(X=[Y,Y,Y,Y], Y=[Y], params={}, do=self.do,dc=self.dc).cree_ix(),
			#
			'o' : DOT1D_2X(X=[X,Y], Y=[Y], params={'activ':1, 'C0':C0}, do=self.do,dc=self.dc).cree_ix(),
			'tanh(c)' : MODULE_i_Activation(X=[Y],   Y=[Y], params={'activ':0}, do=self.do,dc=self.dc).cree_ix(),
			'h'       : MODULE_i_Mul       (X=[Y,Y], Y=[Y], params={         }, do=self.do,dc=self.dc).cree_ix(),
		}

		self.connections = {
			'x' : {0 : None},
			#
			'f' : {0 : ('x', 0), 1 : ('c', -1)},
			'i' : {0 : ('x', 0), 1 : ('c', -1)},
			'u' : {0 : ('x', 0)               },
			#
			'c' : {
				0 : ('f', 0),
				1 : ('c', -1),
				2 : ('i', 0),
				3 : ('u', 0)
			},
			#
			'o' : {0 : ('x', 0), 1 : ('c', 0)},
			'tanh(c)' : {0 : ('c', 0)},
			'h'       : {0 : ('o', 0), 1 : ('tanh(c)', 0)}
		}

		self.cree_elements_connections()
		return self.ix

class LSTM1D_PEEPHOLE__CHAINE(CHAINE):
	img = img_chaine
	bg, fg = 'white', 'black'
	nom = "[LSTM1D_PEEPHOLE] CHAINE"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'H' : 0,
		'N' : 1,
		'C0' : 1
	}
	#	--------------------------
	H       = 'H'
	N       = 'N'
	ELEMENT = LSTM1D_PEEPHOLE


class LSTM1D_PEEPHOLE_PROFOND(Module_Mdl):
	bg, fg = 'light yellow', 'black'
	nom = "[LSTM PROFOND]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'activ' : 0,
		'C0' : 1,
	}