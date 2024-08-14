#! /usr/bin/python3

import struct as st
from random import shuffle
from math import exp

import matplotlib.pyplot as plt

from os import system 

from CONTEXTE import *

######################################################################################

OK = lambda s: print(f"[OK] {s}")

def lire(fichier):
	with open(fichier, 'rb') as co:
		bins = co.read()
		(L,) = st.unpack('I', bins[:4])
		return st.unpack('f'*L, bins[4:])

def softmax(l):
	e = [exp(i) for i in l]
	s = sum(e)
	return [i/s for i in e]

def norme_moyenne(l):
	moyenne = abs(sum(l)/len(l))
	return [e/moyenne for e in l]

def norme(l):
	_min, _max = min(l), max(l)
	return [(e-_min)/(_max - _min) for e in l]

def norme_théorique(l, _min, _max):
	return [(e-_min)/(_max - _min) for e in l]

def norme_relative(l):
	__max = max([abs(min(l)), abs(max(l))])
	_min, _max = -__max, +__max
	return [(e-_min)/(_max - _min) for e in l]

######################################################################################

#python3 -m prixs.dar PRIXS={HEURES} prixs/tester_model_donnee.bin BTC, ETH, ...

from sys import argv
assert len(argv) > (1 + 2)
PRIXS       = int(argv[1].split('=')[1])
fichier_bin = argv[2 ]
MARCHEES    = argv[3:]
print(f'PRIXS={PRIXS}, DEPART={DEPART}, MEGA_T={MEGA_T}, marchés={MARCHEES}')
assert (PRIXS-DEPART-1) % MEGA_T == 0
assert 'BTC' in MARCHEES[0]
assert len(MARCHEES) == 1

marchee = MARCHEES[0]

sources_nom = [
	'prixs', 'low', 'high', 'median',
	'volumes', 'volumes_A', 'volumes_U',
	'jour', 'mois', 'année'
]
sources     = {
	nom_extraction  : lire(f'prixs/{marchee}USDT/{nom_extraction}.bin')
	for nom_extraction in sources_nom
}
assert all(len(v)==PRIXS for k,v in sources.items())

######################################################################################

from prixs.outils import ema, diff, direct, macd, chiffre, rsi, stoch_rsi, diff_ema

#	A faire :
#		1) Introduire les multi entrés
#		2) Tester Drop Out, et Batch Norm
#		3) Cree le premier modèle simple et l'entrainner (Embedeing)
#		4) X.T<L*N> @ W0<N*B> = <B*L>

fonction_ligne = lambda heure, src: {
	'directe'	: direct   (ema(src['prixs'], K=1*heure)),
	'directe4'	: direct   (ema(src['prixs'], K=4*heure)),
#
	'diff'		: diff     (ema(src['prixs'], K=1*heure)),
#
	'diffema'	: diff_ema (ema(src['prixs'], K=1*heure)),
	#
###	'diff'		: diff     (ema(src['prixs'], K=1*heure)),
	'macd1'		: macd     (ema(src['prixs'], K=1*heure), e=1*heure),
	'macd5'		: macd     (ema(src['prixs'], K=4*heure), e=4*heure),
	#
	'ch1k'		: chiffre  (ema(src['prixs'], K=1*heure), __chiffre=1000),
	'ch10k'		: chiffre  (ema(src['prixs'], K=1*heure), __chiffre=10000),
	#
	'rsi14'		: rsi      (ema(src['prixs'], K=1*heure), n=14),
	#
	'st_rsi14'  : stoch_rsi(ema(src['prixs'], K=1*heure), n=14, stoch_n=14),
#	AO
#	%R
#	Chiffre Haut
#	Chiffre Bas
#	'mois'      : direct   (ema(src['mois' ], K=1*heure)),
	#
	'volumes_A' : direct   (ema(src['volumes'  ], K=1*heure)),
	'volumes_U' : direct   (ema(src['volumes_U'], K=1*heure))
}

#	lignes : [{'l0':list, 'l1':list}, ... for _ in INTERVS]
lignes = [fonction_ligne(h, sources) for h in INTERVS]

#	Diff
for i,_ in enumerate(INTERVS):
	for k in lignes[i].keys():
		lignes[i][k] = diff(lignes[i][k])

#	Matplotlib
fig, ax = plt.subplots(len(lignes[0]), len(INTERVS))
for i,h in enumerate(INTERVS):
	for j,(k,v) in enumerate(list(lignes[i].items())):
		ligne = lignes[i][k]
		assert len(ligne) == PRIXS
		ax[j][i].plot(ligne)
		ax[j][i].plot(
			[      PRIXS - i*h - 1  for i in range(N)],
			[ligne[PRIXS - i*h - 1] for i in range(N)]
		)
		ax[j][i].set_title(f'{k} {h}h')
		ax[j][i].get_xaxis().set_visible(False)
plt.show()

######################################################################################

######################################################################################

P = PRIXS
I = len(INTERVS)
L = len(lignes[0])
N = N

print(f"P = {P}")	#	Prixs
print(f"I = {I}")	#	Intervs
print(f"L = {L}")	#	Lignes
print(f"N = {N}")	#	N temps

with open("prixs/lignes_dar.bin", "wb") as co:
	bins = st.pack('IIIII', P,I,L,N, DEPART)
	bins += st.pack('I'*I, *INTERVS)
	for ligne in lignes:
		for k,v in ligne.items():
			bins += st.pack('f'*P, *v)
	co.write(bins)

system("./prixs/cree_dar_bin")
print("## ! Ajouter un BN direct a l'entrée, car les valeurs sont pas normées ! ##")