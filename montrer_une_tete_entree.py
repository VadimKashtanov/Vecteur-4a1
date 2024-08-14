import struct as st

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 1, 1), (0, 0, 0)]
n_bins = 256
cmap_name = 'black_white_purple'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

lire_uints  = lambda I, bins: (st.unpack('I'*I, bins[:4*I]), bins[4*I:])
lire_floats = lambda I, bins: (st.unpack('f'*I, bins[:4*I]), bins[4*I:])

with open("prixs/dar.bin", "rb") as co:
	bins = co.read()

	(T,), bins = lire_uints(1, bins)

	on_va_extraire_quelques_T = 3

	(LIGNES,D,N), bins = lire_uints(3, bins)

	x, bins = lire_floats(LIGNES*N*D*on_va_extraire_quelques_T, bins)
	y, bins = lire_floats(1*on_va_extraire_quelques_T,          bins)

	S = int(LIGNES**.5 + 1)
	fig, ax = plt.subplots(S,S+1+S+1+S)


	for t,depart in [(0,0), (1,S+1), (2,S+1+S+1)]:
		for i in range(LIGNES):
			ax[i % S][depart+int((i - (i % S))/S)].imshow([[x[t*D*N*LIGNES + i*N*D+n*D+d] for n in range(N)] for d in range(D)], cmap=cm)

	plt.show() # Montrer 2-3 entrés differentes