from tkinter_cree_dossier.modules._etc import *

from tkinter_cree_dossier.modules.dot1d import *

class EMBEDE_POSITIONNAL(Module_Mdl):
	bg, fg = 'light blue', 'black'
	nom = "[Embede+Positionnal]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'mot'     : 1,
		'd_model' : 1,
		'mots'    : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		mot     = self.params['mot'    ]
		d_model = self.params['d_model']
		mots    = self.params['mots'   ]

		mot  = int(X / mots)

		#	------------------

		self.elements = {
			'x' : MODULE_i_Y(X=[mot*mots], Y=[mot*mots], params={}).cree_ix(),
			'embede'       : MODULE_i_MatMul_Poid_AP(X=[X], Y=[d_model*mots], params={'Ax':mot, 'Ay':mots, 'Bx':d_model, 'C0':1}).cree_ix(),
			'positionnal'  : MODULE_i_Positionnal   (X=[d_model*mots], Y=[d_model*mots], params={'L':d_model, 'N':mots}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},

			'embede' : {0:('x',0)},
			'positionnal' : {0:('embede',0)}
		}

		self.cree_elements_connections()
		return self.ix

class ANTI_EMBEDE(Module_Mdl):
	bg, fg = 'light blue', 'black'
	nom = "[ANTI EMBEDE]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'dimention' : 1,
		'mots'      : 1,
		'têtes'     : 1,
	}
	def cree_ix(self):
		#	Params
		X = self.X[0]
		Y = self.Y[0]

		dimention = self.params['dimention']
		mots      = self.params['mots'     ]
		têtes     = self.params['têtes'    ]

		Ax        = int(X / mots)

		#	------------------

		self.elements = {
			'x'      : MODULE_i_Y(X=[Ax*mots], Y=[Ax*mots], params={}).cree_ix(),
			'embede' : MODULE_i_MatMul_Poid_AP(X=[X], Y=[dimention*mots], params={'Ax':Ax, 'Ay':mots, 'Bx':dimention, 'C0':1}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},

			'embede' : {0:('x',0)},
		}

		self.cree_elements_connections()
		return self.ix

#	==================================================================

class MultiHeadAttention(Module_Mdl):
	nom = "Scaled Dot Attention : QKV"
	X, Y = [0,0,0], [0]
	X_noms, Y_noms = ["Q", "K", "V"], ["Img"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'clef'    : 1,
		'têtes'   : 1,
	}
	def cree_ix(self):
		#	Params
		Q,K,V = self.X
		Y,    = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots']
		clef    = self.params['clef' ]
		têtes   = self.params['têtes']

		assert Q == K == V == d_model*mots

		QKV = d_model*mots

		self.elements = {
			'Q' : MODULE_i_Canalisation(X=[QKV], Y=[QKV*têtes], params={'C0':têtes}).cree_ix(),
			'K' : MODULE_i_Canalisation(X=[QKV], Y=[QKV*têtes], params={'C0':têtes}).cree_ix(),
			'V' : MODULE_i_Canalisation(X=[QKV], Y=[QKV*têtes], params={'C0':têtes}).cree_ix(),

			'q' : MODULE_i_MatMul_Poid_AP(X=[QKV*têtes], Y=[clef*mots*têtes], params={'Ax':d_model, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),
			'k' : MODULE_i_MatMul_Poid_AP(X=[QKV*têtes], Y=[clef*mots*têtes], params={'Ax':d_model, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),
			'v' : MODULE_i_MatMul_Poid_AP(X=[QKV*têtes], Y=[clef*mots*têtes], params={'Ax':d_model, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),

			'q@k.T' :            MODULE_i_QKtDivClef(X=[clef*mots*têtes,  clef*mots*têtes], Y=[mots *mots*têtes], params={'Ax':clef, 'Ay':mots, 'Bx':mots, 'C0':têtes}).cree_ix(),
			'softmax(q@k.T)' :   MODULE_i_Softmax   (X=[mots *mots*têtes                 ], Y=[mots *mots*têtes], params={'Vect':mots},                               ).cree_ix(),
			'softmax(q@k.T)@v' : MODULE_i_MatMul    (X=[mots *mots*têtes, clef*mots*têtes], Y=[clef*mots*têtes ], params={'Ax':mots, 'Ay':mots, 'Bx':clef, 'C0':têtes}).cree_ix(),

			'concatenation' : MODULE_i_Concatenation(X=[clef*mots*têtes], Y=[(clef*têtes)*mots], params={'Ax':clef, 'Ay':mots, 'Ay-c0':têtes, 'C0':1}).cree_ix(),
			
			'lineair' : MODULE_i_MatMul_Poid_AP(X=[(clef*têtes)*mots], Y=[(d_model)*mots], params={'Ax':clef*têtes, 'Ay':mots, 'Bx':d_model, 'C0':1}).cree_ix(),
		}

		self.connections = {
			'Q' : {0:None},
			'K' : {0:None},
			'V' : {0:None},
			#
			'q' : {0:('Q',0)},
			'k' : {0:('K',0)},
			'v' : {0:('V',0)},
			#
			#'k.T' : {0:('k',0)},
			#
			'q@k.T'            : {0:('q',0), 1:('k',0)},
			'softmax(q@k.T)'   : {0:('q@k.T', 0)},
			'softmax(q@k.T)@v' : {0:('softmax(q@k.T)', 0), 1 : ('v', 0)},
			#
			'concatenation' : {0:('softmax(q@k.T)@v', 0)},
			#
			'lineair' : {0:('concatenation',0)}
		}

		self.cree_elements_connections()
		return self.ix

class Self_MultiHeadAttention(Module_Mdl):
	nom = "Self Scaled Dot Attention : QKV"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'clef'    : 1,
		'têtes'   : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots']
		clef    = self.params['clef' ]
		têtes   = self.params['têtes']

		self.elements = {
			'x' : MODULE_i_Y(X=[X], Y=[X], params={}).cree_ix(),
			'multihead' : MultiHeadAttention(X=[X,X,X], Y=[Y], params={
				'd_model':d_model,
				'mots':mots,
				'clef':clef,
				'têtes':têtes
				}).cree_ix(),
		}
		self.connections = {
			'x' : {0:None},
			'multihead' : {0:('x',0), 1:('x',0), 2:('x',0)}
		}

		self.cree_elements_connections()
		return self.ix

##############################################################

class FFN(Module_Mdl):
	nom = "FFN : Gelu(x@P+b)@P+b"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'ff'      : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots']
		ff      = self.params['ff']

		self.elements = {
			'x' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			#
			'x@P' : MODULE_i_MatMul_Poid_AP(X=[d_model*mots], Y=[ff*mots], params={'Ax':d_model, 'Ay':mots, 'Bx':ff, 'C0':1}).cree_ix(),
			'Gelu(x@P+b)' : MODULE_i_Activation_Poid(X=[ff*mots], Y=[ff*mots], params={'activ':f_GELU}).cree_ix(),
			'Gelu(x@P+b)@P' : MODULE_i_MatMul_Poid_AP(X=[ff*mots], Y=[d_model*mots], params={'Ax':ff, 'Ay':mots, 'Bx':d_model, 'C0':1}).cree_ix(),
			'Gelu(x@P+b)@P+b' : MODULE_i_Activation_Poid(X=[d_model*mots], Y=[d_model*mots], params={'activ':f_GELU}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},
			#
			'x@P' : {0:('x',0)},
			'Gelu(x@P+b)' : {0:('x@P',0)},
			'Gelu(x@P+b)@P' : {0:('Gelu(x@P+b)',0)},
			'Gelu(x@P+b)@P+b' : {0:('Gelu(x@P+b)@P',0)},
		}

		self.cree_elements_connections()
		return self.ix

class Contexte(Module_Mdl):
	nom = "x[t=0] -> repetition"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots'   ]

		#mot0 -> mot0
		#mot1 -> mot0
		#mot2 -> mot0

		self.elements = {
			'x' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			#
			'x[t=0]' : MODULE_i_Select_Vect (X=[d_model*mots], Y=[d_model], params={'Vect':d_model, 'N':0}).cree_ix(),
			'canale' : MODULE_i_Canalisation(X=[d_model], Y=[d_model*mots], params={'C0':mots}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},
			#
			'x[t=0]' : {0:('x',0)},
			'canale' : {0:('x[t=0]',0)}
		}

		self.cree_elements_connections()
		return self.ix

class FFN_Contexte(Module_Mdl):
	nom = "FFN : x union ctx*I"
	X, Y = [0,0], [0]
	X_noms, Y_noms = ["X", "CXT"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'ff'      : 1,
		'i-cxt'   : 1,
	}
	def cree_ix(self):
		#	Params
		X,CXT = self.X
		Y,    = self.Y

		d_model = self.params['d_model']
		mots    = self.params['mots']
		ff      = self.params['ff'   ]
		i_cxt   = self.params['i-cxt']

		assert CXT == i_cxt * mots * d_model

		self.elements = {
			'x' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			#
			'x_cxt_pile'    : MODULE_i_Union(X=[d_model*mots, CXT], Y=[d_model*mots+CXT]).cree_ix(),
			'x_cxt_concate' : MODULE_i_Concatenation(X=[d_model*mots+CXT], Y=[(d_model+i_cxt*d_model)*mots], params={'Ax':d_model, 'Ay':mots, 'Ay-c0':1+i_cxt, 'C0':1}).cree_ix(),
			#
			'x@P' : MODULE_i_MatMul_Poid_AP(X=[(d_model+i_cxt*d_model)*mots], Y=[ff*mots], params={'Ax':d_model+i_cxt*d_model, 'Ay':mots, 'Bx':ff, 'C0':1}).cree_ix(),
			'Gelu(x@P+b)' : MODULE_i_Activation_Poid(X=[ff*mots], Y=[ff*mots], params={'activ':f_GELU}).cree_ix(),
			'Gelu(x@P+b)@P' : MODULE_i_MatMul_Poid_AP(X=[ff*mots], Y=[d_model*mots], params={'Ax':ff, 'Ay':mots, 'Bx':d_model, 'C0':1}).cree_ix(),
			'Gelu(x@P+b)@P+b' : MODULE_i_Activation_Poid(X=[d_model*mots], Y=[d_model*mots], params={'activ':f_GELU}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},
			#
			'x_cxt_pile' : {0:('x',0), 1:None},
			'x_cxt_concate' : {0:('x_cxt_pile',0)},
			#
			'x@P' : {0:('x_cxt_concate',0)},
			'Gelu(x@P+b)' : {0:('x@P',0)},
			'Gelu(x@P+b)@P' : {0:('Gelu(x@P+b)',0)},
			'Gelu(x@P+b)@P+b' : {0:('Gelu(x@P+b)@P',0)},
		}

		self.cree_elements_connections()
		return self.ix

class ENCODEUR(Module_Mdl):
	bg, fg = 'light blue', 'black'
	nom = "[ENCODEUR]"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'têtes'   : 1,
		'clef'    : 1,
		'ff'      : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots = self.params['mots']
		têtes = self.params['têtes']
		clef = self.params['clef']
		ff = self.params['ff']

		#	------------------

		self.elements = {
			'x' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			#
			'MultiHeadAttention' : Self_MultiHeadAttention(X=[d_model*mots], Y=[d_model*mots], params={'d_model' : d_model,'mots' : mots, 'clef' : clef, 'têtes' : têtes}).cree_ix(),
			#
			'Somme 0' : MODULE_i_Somme(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			'Norme 0' : BATCH_NORM    (X=[d_model*mots], Y=[d_model*mots], params={'C0':mots}).cree_ix(),
			#
			'FFN' : FFN(X=[d_model*mots], Y=[d_model*mots], params={'d_model':d_model, 'mots':mots, 'ff':ff}).cree_ix(),
			#
			'Somme 1' : MODULE_i_Somme(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			'Norme 1' : BATCH_NORM    (X=[d_model*mots], Y=[d_model*mots], params={'C0':mots}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},
			#
			'MultiHeadAttention' : {0:('x',0)},
			#
			'Somme 0' : {0:('MultiHeadAttention',0), 1:('x',0)},
			'Norme 0' : {0:('Somme 0',0)},
			#
			'FFN' : {0:('Norme 0',0)},
			#
			'Somme 1' : {0:('FFN',0), 1:('Norme 0',0)},
			'Norme 1' : {0:('Somme 1',0)},
		}

		self.cree_elements_connections()
		return self.ix

class ENCODEUR_CHAINE(Module_Mdl):
	img = img_chaine
	bg, fg = 'light blue', 'black'
	nom = "[ENCODEUR] Chaine"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'têtes'   : 1,
		'clef'    : 1,
		'ff'      : 1,
		#
		'N' : 1,
	}
	def cree_ix(self):
		#	Params
		X, = self.X
		Y, = self.Y

		d_model = self.params['d_model']
		mots = self.params['mots']
		têtes = self.params['têtes']
		clef = self.params['clef']
		ff = self.params['ff']
		N = self.params['N']

		self.elements = {**{
			'-1' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
		}, **{
			f'{i}' : ENCODEUR(X=[d_model*mots],Y=[d_model*mots], params={
				'd_model' : d_model,
				'mots'    : mots,
				'têtes'   : têtes,
				'clef'    : clef,
				'ff'      : ff,
				}).cree_ix()
			for i in range(N)
		}}
		self.connections = {**{
			'-1' : {0:None},
		}, **{
			f'{i}' : {0:(f'{i-1}',0)}
			for i in range(N)
		}}
		self.cree_elements_connections()
		return self.ix

class CONTEXTE_ENCODEUR(Module_Mdl):
	bg, fg = 'light blue', 'black'
	nom = "[CONTEXTE ENCODEUR]"
	X, Y = [0,0], [0]
	X_noms, Y_noms = ["X", "CXT"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'têtes'   : 1,
		'clef'    : 1,
		'ff'      : 1,
		'i-cxt'   : 1,
	}
	def cree_ix(self):
		#	Params
		X,CXT = self.X
		Y,    = self.Y

		d_model = self.params['d_model']
		mots = self.params['mots']
		têtes = self.params['têtes']
		clef = self.params['clef']
		ff = self.params['ff']
		i_cxt = self.params['i-cxt']

		#	------------------

		self.elements = {
			'x' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			'cxt' : MODULE_i_Y(X=[CXT], Y=[CXT], params={}).cree_ix(),
			#
			'MultiHeadAttention' : Self_MultiHeadAttention(X=[d_model*mots], Y=[d_model*mots], params={'d_model' : d_model,'mots' : mots, 'clef' : clef, 'têtes' : têtes}).cree_ix(),
			#
			'Somme 0' : MODULE_i_Somme(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={}         ).cree_ix(),
			'Norme 0' : BATCH_NORM    (X=[d_model*mots], Y=[d_model*mots], params={'C0':mots}).cree_ix(),
			#
			'FFN' : FFN_Contexte(X=[d_model*mots,CXT], Y=[d_model*mots], params={'d_model':d_model, 'mots':mots, 'ff':ff, 'i-cxt':i_cxt}).cree_ix(),
			#
			'Somme 1' : MODULE_i_Somme(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={}          ).cree_ix(),
			'Norme 1' : BATCH_NORM    (X=[d_model*mots], Y=[d_model*mots], params={'C0':mots}).cree_ix(),
		}

		self.connections = {
			'x' : {0:None},
			'cxt' : {0:None},
			#
			'MultiHeadAttention' : {0:('x',0)},
			#
			'Somme 0' : {0:('MultiHeadAttention',0), 1:('x',0)},
			'Norme 0' : {0:('Somme 0',0)},
			#
			'FFN' : {0:('Norme 0',0), 1:('cxt',0)},
			#
			'Somme 1' : {0:('FFN',0), 1:('Norme 0',0)},
			'Norme 1' : {0:('Somme 1',0)},
		}

		self.cree_elements_connections()
		return self.ix

class CONTEXTE_ENCODEUR_CHAINE(Module_Mdl):
	img = img_chaine
	bg, fg = 'light blue', 'black'
	nom = "[CONTEXTE ENCODEUR] Chaine"
	X, Y = [0,0], [0]
	X_noms, Y_noms = ["X", "CXT"], ["Y"]
	params = {
		'd_model' : 1,
		'mots'    : 1,
		'têtes'   : 1,
		'clef'    : 1,
		'ff'      : 1,
		'i-cxt'   : 1,
		#
		'N' : 1,
	}
	def cree_ix(self):
		#	Params
		X,CXT = self.X
		Y,    = self.Y

		d_model = self.params['d_model']
		mots = self.params['mots']
		têtes = self.params['têtes']
		clef = self.params['clef']
		ff = self.params['ff']
		i_cxt = self.params['i-cxt']
		N = self.params['N']

		self.elements = {**{
			'-1' : MODULE_i_Y(X=[d_model*mots], Y=[d_model*mots], params={}).cree_ix(),
			'cxt' : MODULE_i_Y(X=[CXT], Y=[CXT], params={}).cree_ix(),
		}, **{
			f'{i}' : CONTEXTE_ENCODEUR(X=[d_model*mots,CXT],Y=[d_model*mots], params={
				'd_model' : d_model,
				'mots'    : mots,
				'têtes'   : têtes,
				'clef'    : clef,
				'ff'      : ff,
				'i-cxt'   : i_cxt,
				}).cree_ix()
			for i in range(N)
		}}
		self.connections = {**{
			'-1' : {0:None},
			'cxt' : {0:None},
		}, **{
			f'{i}' : {0:(f'{i-1}',0), 1:('cxt',0)}
			for i in range(N)
		}}
		self.cree_elements_connections()
		return self.ix