from tkinter_cree_dossier.modules._etc import *

from tkinter_cree_dossier.modules.unions import *

from tkinter_cree_dossier.modules.attention import *

class MODELE_TRANSFORMER_4(Module_Mdl):
	bg, fg = 'purple', 'white'
	nom = "Modèle Transformeur 4 (Contexte)"
	X, Y = [4*12*64], [4*12*64]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'Données : intervalles' :   4,
		'Données : taille_mot ' :  12,	#	1 pixel par ligne
		'Données : mots       ' :  64,	#	temps : 64
		#
		'Modèle  : d_model    ' : 128,
		'Modèle  : clef       ' :  16,
		'Modèle  : têtes      ' :   8,
		'Modèle  : ff         ' : 256,
		#
		'Chaines : N          ' :   6,
	}
	def cree_ix(self):
		#	Params
		X, Y = self.X[0], self.Y[0]

		intervs = self.params['Données : intervalles']
		mot     = self.params['Données : taille_mot ']	#mot
		mots    = self.params['Données : mots       ']	#mots
		#
		d_model = self.params['Modèle  : d_model    ']
		clef    = self.params['Modèle  : clef       ']
		têtes   = self.params['Modèle  : têtes      ']
		ff      = self.params['Modèle  : ff         ']
		#
		N       = self.params['Chaines : N          ']

		assert intervs == 4

		self.elements = {
			'x' : MODULE_i_Y(X=[intervs*mot*mots], Y=[intervs*mot*mots], params={}).cree_ix(),

			#	=== Encodeur Chaque Intervalle ===

			#	Intervalle #3
			'select           3' : MODULE_i_Select_Vect(X=[intervs*mot*mots], Y=[mot*mots],     params={'Vect':mot*mots, 'N':3}).cree_ix(),
			'embede&position  3' : EMBEDE_POSITIONNAL  (X=[mot*mots],         Y=[d_model*mots], params={'mot':mot, 'd_model':d_model,'mots':mots}).cree_ix(),
			'chaine encodeur  3' : ENCODEUR_CHAINE     (X=[d_model*mots],     Y=[d_model*mots], params={'d_model' : d_model, 'mots' : mots, 'têtes' : têtes, 'clef' : clef, 'ff' : ff, 'N' : N}).cree_ix(),
			'Dernier Contexte 3' : Contexte            (X=[d_model*mots],     Y=[d_model*mots], params={'d_model':d_model, 'mots':mots}).cree_ix(),

			#	Intervalle #2
			'select           2' : MODULE_i_Select_Vect    (X=[intervs*mot*mots],          Y=[mot*mots],     params={'Vect':mot*mots, 'N':2}).cree_ix(),
			'embede&position  2' : EMBEDE_POSITIONNAL      (X=[mot*mots],                  Y=[d_model*mots], params={'mot':mot, 'd_model':d_model,'mots':mots}).cree_ix(),
			'chaine encodeur  2' : CONTEXTE_ENCODEUR_CHAINE(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={'d_model' : d_model, 'mots' : mots, 'têtes' : têtes, 'clef' : clef, 'ff' : ff, 'i-cxt' : 1, 'N' : N}).cree_ix(),
			'Dernier Contexte 2' : Contexte                (X=[d_model*mots],              Y=[d_model*mots], params={'d_model':d_model, 'mots':mots}).cree_ix(),

			#	Intervalle #1
			'select           1' : MODULE_i_Select_Vect    (X=[intervs*mot*mots],          Y=[mot*mots],     params={'Vect':mot*mots, 'N':1}).cree_ix(),
			'embede&position  1' : EMBEDE_POSITIONNAL      (X=[mot*mots],                  Y=[d_model*mots], params={'mot':mot, 'd_model':d_model,'mots':mots}).cree_ix(),
			'chaine encodeur  1' : CONTEXTE_ENCODEUR_CHAINE(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={'d_model' : d_model, 'mots' : mots, 'têtes' : têtes, 'clef' : clef, 'ff' : ff, 'i-cxt' : 1, 'N' : N}).cree_ix(),
			'Dernier Contexte 1' : Contexte                (X=[d_model*mots],              Y=[d_model*mots], params={'d_model':d_model, 'mots':mots}).cree_ix(),

			#	Intervalle #0
			'select           0' : MODULE_i_Select_Vect    (X=[intervs*mot*mots],          Y=[mot*mots],     params={'Vect':mot*mots, 'N':0}).cree_ix(),
			'embede&position  0' : EMBEDE_POSITIONNAL      (X=[mot*mots],                  Y=[d_model*mots], params={'mot':mot, 'd_model':d_model,'mots':mots}).cree_ix(),
			'chaine encodeur  0' : CONTEXTE_ENCODEUR_CHAINE(X=[d_model*mots,d_model*mots], Y=[d_model*mots], params={'d_model' : d_model, 'mots' : mots, 'têtes' : têtes, 'clef' : clef, 'ff' : ff, 'i-cxt' : 1, 'N' : N}).cree_ix(),

			#	=== Union Vers i+1 bloque des 4 intervalles (Sortie) ===

			'anti embede interv= 1' : MODULE_i_MatMul_Poid_AP(X=[d_model*mots], Y=[mot*mots], params={'Ax':d_model,'Ay':mots,'Bx':mot,'C0':1}).cree_ix(),
			'anti embede interv= 4' : MODULE_i_MatMul_Poid_AP(X=[d_model*mots], Y=[mot*mots], params={'Ax':d_model,'Ay':mots,'Bx':mot,'C0':1}).cree_ix(),
			'anti embede interv=16' : MODULE_i_MatMul_Poid_AP(X=[d_model*mots], Y=[mot*mots], params={'Ax':d_model,'Ay':mots,'Bx':mot,'C0':1}).cree_ix(),
			'anti embede interv=64' : MODULE_i_MatMul_Poid_AP(X=[d_model*mots], Y=[mot*mots], params={'Ax':d_model,'Ay':mots,'Bx':mot,'C0':1}).cree_ix(),

			'union sortie' : UNION_4(X=[mot*mots,mot*mots,mot*mots,mot*mots], Y=[4*mot*mots], params={}).cree_ix(),
		}
		self.connections = {
			'x' : {0:None},

			'select           3' : {0:('x',0)},
			'embede&position  3' : {0:('select           3',0)},
			'chaine encodeur  3' : {0:('embede&position  3',0)},
			'Dernier Contexte 3' : {0:('chaine encodeur  3',0)},

			#	Intervalle #2
			'select           2' : {0:('x',0)},
			'embede&position  2' : {0:('select           2',0)},
			'chaine encodeur  2' : {0:('embede&position  2',0), 1:('Dernier Contexte 3',0)},
			'Dernier Contexte 2' : {0:('chaine encodeur  2',0)},

			#	Intervalle #1
			'select           1' : {0:('x',0)},
			'embede&position  1' : {0:('select           1',0)},
			'chaine encodeur  1' : {0:('embede&position  1',0), 1:('Dernier Contexte 2',0)},
			'Dernier Contexte 1' : {0:('chaine encodeur  1',0)},

			#	Intervalle #0
			'select           0' : {0:('x',0)},
			'embede&position  0' : {0:('select           0',0)},
			'chaine encodeur  0' : {0:('embede&position  0',0), 1:('Dernier Contexte 1',0)},

			#	=== Union Vers i+1 bloque des 4 intervalles (Sortie) ===

			'anti embede interv= 1' : {0:('chaine encodeur  0',0)},
			'anti embede interv= 4' : {0:('chaine encodeur  1',0)},
			'anti embede interv=16' : {0:('chaine encodeur  2',0)},
			'anti embede interv=64' : {0:('chaine encodeur  3',0)},

			'union sortie' : {
				0:('anti embede interv= 1',0),
				1:('anti embede interv= 4',0),
				2:('anti embede interv=16',0),
				3:('anti embede interv=64',0),
			}
		}

		self.cree_elements_connections()
		return self.ix

class MODELE_TRANSFORMER_4_3_2_1(Module_Mdl):
	bg, fg = 'purple', 'white'
	nom = "Modèle Transformeur 4->3->2->1"
	X, Y = [0], [0]
	X_noms, Y_noms = ["X"], ["Y"]
	params = {
		'intervalles' : 4,
		'taille__mot' : 12,	#	1 pixel par ligne
		'nb_____mots' : 64,	#	temps : 64
		#
		'd-modèle' : 1,
		'têtes'    : 1,
	}