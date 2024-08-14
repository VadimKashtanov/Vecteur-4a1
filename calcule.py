from os import system
import struct as st

from CONTEXTE import *

def calcule(donnees, NOM, MEGA_T):
	prixs = [float(c) for _,o,h,l,c,vB,vU in donnees]

	system(f"python3 -m prixs.ecrire_multi_sources prixs/{NOM}USDT.csv")

	system(f"python3 -m prixs.dar PRIXS={len(prixs)} prixs/tester_model_donnee.bin bitgetBTC")

	system("rm les_predictions.bin")

	####################################################################################

	system(f"./prog_tester_le_mdl")

	with open("les_predictions.bin", 'rb') as co:
		bins = co.read()
		I = int( int(len(bins)/4) / 2)
		les_predictions = st.unpack('f'*I, bins[0*4*I:1*4*I])
		les_delats      = st.unpack('f'*I, bins[1*4*I:2*4*I])

	return les_predictions, les_delats