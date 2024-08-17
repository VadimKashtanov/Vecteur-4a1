import struct as st
import matplotlib.pyplot as plt

ETAPE = lambda I, intitulé: print(f"\033[92m[OK]\033[0m Etape {I}: {intitulé}")

def normer(l):
	_min, _max = min(l), max(l)
	return [(e-_min)/(_max-_min) for e in l]

def normer_11(l):
	_min, _max = min(l), max(l)
	return [2*(e-_min)/(_max-_min)-1 for e in l]

__sng = lambda x: (1 if x > 0 else -1)

####################################################################################

from CONTEXTE import *

d = "1H" # "15m"

T = 24*5#4*7*24 #30 * 24

assert T % MEGA_T == 0

HEURES = DEPART + T# + 1

from données_bitget_2 import DONNEES_BITGET, faire_un_csv

donnees = DONNEES_BITGET(HEURES, d)
csv     = faire_un_csv(donnees, NOM="bitgetBTCUSDT")

print(f"Len donnees = {len(donnees)}")

with open('prixs/bitgetBTCUSDT.csv', 'w') as co:
	co.write(csv)

ETAPE(1, "Ecriture CSV")

####################################################################################

from calcule import calcule

les_predictions, cuda_les_delats, cuda_les_prixs = calcule(donnees, "bitgetBTC", MEGA_T)

####################################################################################

prixs = [float(c) for _,o,h,l,c,vB,vU in donnees]

python_prixs  = prixs[DEPART:]
python_deltas = [(python_prixs[i+1]/python_prixs[i] - 1 if i!=len(python_prixs)-1 else 0.0) for i in range(len(python_prixs))]

flotant = lambda f: '{a:+.7f}'.format(a=f)
print()
print("cuda   : prixs ", list(map(flotant, cuda_les_prixs [-10:])))
print("python : prixs ", list(map(flotant, python_prixs   [-10:])))
print()
print("les_predictions", list(map(flotant, les_predictions[-10:])))
print()
print("cuda   : delta ", list(map(flotant, cuda_les_delats[-10:])))
print("python : delta ", list(map(flotant, python_deltas  [-10:])))
print()

####################################################################################

print(len(python_prixs), len(les_predictions))
assert len(python_prixs) == len(les_predictions)

a =          (les_predictions)
b = normer_11(python_prixs   )

deltas_normés = [0] + [abs(b[i+1]-b[i]) for i in range(len(b)-1)]

for i in range(int(len(les_predictions)/MEGA_T)):
	#	Ligne |
	plt.plot([len(les_predictions) - i*MEGA_T]*2, [-1, 1])


	#	Les courbes
	les_p1p0  = deltas_normés[i*MEGA_T:(i+1)*MEGA_T]
	les_preds =             a[i*MEGA_T:(i+1)*MEGA_T]

	s=0
	courbe = [b[i*MEGA_T] + (s:=( s+__sng(elm)*les_p1p0[j] )) for j,elm in enumerate(a[i*MEGA_T:(i+1)*MEGA_T])]

	#	X
	x = list(range(i*MEGA_T, (i+1)*MEGA_T))

	if i == 0:
		plt.plot(x, les_preds, 'm-o', label='o = Predictions')
		plt.plot(x, courbe, 'g', label='Petite tendance')
	else:
		plt.plot(x, les_preds, 'm-o')
		plt.plot(x, courbe, 'g')

		u = [1]
		for j in range(len(les_preds)-1):
			p1 = 2+b[i*MEGA_T+j+1]
			p0 = 2+b[i*MEGA_T+j+0]
			
			u += [ u[-1] + u[-1] *20* ( p1/p0 - 1) *les_preds[j] ]
			if (u[-1]<0): u[-1]=0
		plt.plot(x, [_u-2 for _u in u])

	#	Fleches
	for j in range(MEGA_T):
		if a[i*MEGA_T+j] >= 0.0:
			plt.plot([i*MEGA_T+j, i*MEGA_T+j], [b[i*MEGA_T+j], b[i*MEGA_T+j] + 0.03], 'g')
		else:
			plt.plot([i*MEGA_T+j, i*MEGA_T+j], [b[i*MEGA_T+j], b[i*MEGA_T+j] - 0.03], 'r')

plt.plot(b, 'c-^', label='prix')
plt.legend()
plt.show()

####################################################################################

print(f"len(les_predictions) = {len(les_predictions)}")
print(f"len(prixs)           = {len(python_prixs)}   ")

fig, ax = plt.subplots(2)

LEVIERS = [10, 25, 50, 125]

#
for L in LEVIERS:
	u = 100
	_u0 = []
	#La dernière pred est pour dans 1h (et on sait pas encore le prixs dans 1h)
	for i in range(len(les_predictions) - 1):
		u += u * L * les_predictions[i] * (python_prixs[i+1]/python_prixs[i]-1)
		_u0 += [u]
		if u < 0: u = 0
	#
	ax[0].plot(
		_u0,
		label='*'+str(L)
	)
ax[0].legend()
#
for L in LEVIERS:
	u = 100
	_u0 = []
	#La dernière pred est pour dans 1h (et on sait pas encore le prixs dans 1h)
	for i in range(len(les_predictions) - 1):
		u += u * L * __sng(les_predictions[i]) * (python_prixs[i+1]/python_prixs[i]-1)
		_u0 += [u]
		if u < 0: u = 0
	#
	ax[1].plot(
		_u0,
		label='sng(p)*'+str(L)
	)
ax[1].legend()
#
plt.show()