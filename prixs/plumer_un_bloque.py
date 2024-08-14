import struct as st
import matplotlib.pyplot as plt
from time import sleep

print("plumer_un_bloque.py ...")
with open('prixs/dar.bin', 'rb') as co:
	bins = co.read()
	#
	I,T,L,N, = st.unpack('IIII', bins[:4*4])
	#
	#dar = st.unpack('f'*I*T*L*N, bins[4*4:])
	#
	t = int(0.99 * T)
	#
	
	#	Plumer les Bloques de la chaine
	fig, ax = plt.subplots(1, I)
	for i in range(I):
		depart = i*(T*L*N) + t*(L*N)
		bloque = st.unpack('f'*L*N, bins[4*4+4*depart:4*4+4*depart+4*L*N])
		#
		ax[i].imshow([[bloque[n*L + l] for l in range(L)] for n in range(N)])
	plt.show()

	#	Plumer les courbes
	fig, ax = plt.subplots(L, I)
	for i in range(I):
		depart = i*(T*L*N) + t*(L*N)
		bloque = st.unpack('f'*L*N, bins[4*4+4*depart:4*4+4*depart+4*L*N])
		for l in range(L):	#	.T
			ax[l][i].plot([bloque[n*L+l] for n in range(N)])
	plt.show()