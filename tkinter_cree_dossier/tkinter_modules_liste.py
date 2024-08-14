from tkinter_cree_dossier.modules._etc import *

from tkinter_cree_dossier.modules.unions import *
from tkinter_cree_dossier.modules.selects import *

from tkinter_cree_dossier.modules.dot1d   import *
from tkinter_cree_dossier.modules.dot1d_2 import *
from tkinter_cree_dossier.modules.dot1d_3 import *

from tkinter_cree_dossier.modules.dot1d_reccurent import *

from tkinter_cree_dossier.modules.dot2d import *

from tkinter_cree_dossier.modules.lstm import *
from tkinter_cree_dossier.modules.lstm_peephole import *

from tkinter_cree_dossier.modules.signale_logistique import *

from tkinter_cree_dossier.modules.attention import *
from tkinter_cree_dossier.modules.transformer_prixs import *

class NONE(Module_Mdl):
	nom = ""
	X,      Y      = [], []
	X_noms, Y_noms = [], []
	params = {}
	def cree_ix(self):
		return []

#	=========================================================================	#

modules = [
	DOT1D,
	DOT1D__CHAINE,
	DOT1D_2X,
	DOT1D_3X,
	DOT2D_AP,
	DOT2D_PA,
	NONE,
	NONE,
	NONE,
#	-----------------	#
#	DOT1D_RECCURENT,
#	DOT1D_RECCURENT__CHAINE,
#	NONE,
#	DOT1D_RECCURENT_N,
#	DOT1D_RECCURENT_N__CHAINE,
#	NONE,
#	NONE,
#	NONE,
#	NONE,
#	-----------------	#
#	LSTM1D_2X,
#	LSTM1D_3X,
#	LSTM1D_PROFOND,
#	NONE,
#	LSTM1D_PEEPHOLE,
#	LSTM1D_PEEPHOLE__CHAINE,
#	NONE,
#	SIGNALE_LOGISTIQUE,
#	NONE,
#	----------------	#
	NONE,
	ENCODEUR,
	ENCODEUR_CHAINE,
	NONE,
	CONTEXTE_ENCODEUR,
	CONTEXTE_ENCODEUR_CHAINE,
	NONE,
	EMBEDE_POSITIONNAL,
	ANTI_EMBEDE,
#	----------------	#
	SOMME3,
	NONE,
	NORMALISATION,
	NONE,
	BATCH_NORM,
	NONE,
	NONE,
	NONE,
	Self_MultiHeadAttention,
#	----------------	#
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	UNION_3,
	UNION_4,
	NONE,
	SPLIT_4,
#	----------------	#
	MODELE_TRANSFORMER_4,
	MODELE_TRANSFORMER_4_3_2_1,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
]

"""
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
	NONE,
"""