#! /usr/bin/python3

from random import random
import struct as st

from CONTEXTE import *

rnd = lambda : 2*random()-1

ema = lambda l,K,ema=0: [ema:=(ema*(1-1/K) + e/K) for e in l]

prixs      = lambda infos:     [p                  for j,m,a, p,vb,vu,l,h,t in infos]
_low       = lambda infos:     [l                  for j,m,a, p,vb,vu,l,h,t in infos]
_high      = lambda infos:     [h                  for j,m,a, p,vb,vu,l,h,t in infos]
median     = lambda infos:     [(h+l)/2            for j,m,a, p,vb,vu,l,h,t in infos]
volumes    = lambda infos,s=0: [s:=(s + (vb*p-vu)) for j,m,a, p,vb,vu,l,h,t in infos]
volumes_A  = lambda infos:     [vb                 for j,m,a, p,vb,vu,l,h,t in infos]
volumes_U  = lambda infos:     [vu                 for j,m,a, p,vb,vu,l,h,t in infos]
tradecount = lambda infos:     [t                  for j,m,a, p,vb,vu,l,h,t in infos]
_jour      = lambda infos:     [j                  for j,m,a, p,vb,vu,l,h,t in infos]
_mois      = lambda infos:     [m                  for j,m,a, p,vb,vu,l,h,t in infos]
_année     = lambda infos:     [a                  for j,m,a, p,vb,vu,l,h,t in infos]


def bruit_pro_normalisant(l):
	#return l
	_min = min(l)
	_max = max(l)
	alea = abs(_max-_min) * 0.000001
	return [i+alea*rnd() for i in l]

Jour  = lambda d: float(d.split(' ')[1].split(':')[0])/24 #quelle période du Jour
Mois  = lambda d: float(d.split(' ')[0].split('-')[2])/31 #quelle période du Mois
Année = lambda d: float(d.split(' ')[0].split('-')[1])/12 #quelle période de l'année

def lire_une_source(source):
	with open(source, "r") as co:
		text = co.read().split('\n')
		del text[ 0]
		del text[ 0]
		del text[-1]
		lignes = [l.split(',') for l in text][::-1] # Du plus recent -> Du plus ancien
		#
		NOM    = lignes[0][2]
		infos  = [
			(
				Jour(Date),
				Mois(Date),
				Année(Date),
				float(Close),
				float(Volume_BTC),
				float(Volume_USDT),
				float(Low),
				float(High),
				float(int(tradecount))
			)
			for Unix,Date,Symbol,Open,High,Low,Close,Volume_BTC,Volume_USDT,tradecount in lignes
		]
		#
	return infos, NOM

#python3 prixs/ecrire_multi_sources.py prixs/BTC.csv prixs/bitgetBTCUSDT.csv ...

if __name__ == "__main__":
	from sys import argv

	sources        = [argv[i] for i in range(1, len(argv))]

	sources_lignes = [lire_une_source(source) for source in sources]

	sorties = {
		f'prixs/{NOM}/{nom_extraction}.bin' : bruit_pro_normalisant(exctration(lignes))
		for lignes,NOM in sources_lignes
			for exctration, nom_extraction in [
				(prixs,      'prixs'  ),
				( _low,      'low'    ),
				(_high,      'high'   ),
				(median,     'median' ),
				(volumes,    'volumes'  ),
				(volumes_A,  'volumes_A'),
				(volumes_U,  'volumes_U'),
				#(tradecount, 'tradecount'),
				(_jour,      'jour' ),
				(_mois,      'mois' ),
				(_année,     'année'),
		]
	}

	for fichier, liste in sorties.items():
		with open(fichier, "wb") as co:
			print(f'LEN {fichier}   = {len(liste)}')
			co.write(st.pack('I', len(liste)))
			co.write(st.pack('f'*len(liste), *liste))