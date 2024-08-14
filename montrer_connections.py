#! /usr/bin/python3

import matplotlib.pyplot as plt
import struct as st
from time import sleep

from tkinter_cree_dossier.tkinter_insts import liste_insts

lire_uints = lambda I, bins: (st.unpack('I'*I, bins[:4*I]), bins[4*I:])

INSTS = []

with open("mdl.st.bin", "rb") as co:
	bins = co.read()

	(insts,), bins = lire_uints(1, bins)

	for inst in range(insts):
		(ID,), bins = lire_uints(1, bins)

		xs = len(liste_insts[ID].X     )
		ps = len(liste_insts[ID].params)

		print(f'ID={ID} xs={xs} ps={ps}')

		INSTS += [[]]

		for x in range(xs):
			(X,x,xt), bins  = lire_uints(3, bins)
			INSTS[-1] += [(X,x,xt)]

		(y,), bins = lire_uints(1, bins)

		params, bins = lire_uints(ps, bins)

		(do,dc), bins = lire_uints(2, bins)

	(entree, sortie), bins = lire_uints(2, bins)

for l in INSTS:
	print(l)

bloques = [[i] for i in range(insts)]

trouver = lambda bloques,i: [(l,bloques[l].index(i)) for l in range(len(bloques)) if i in bloques[l]][0]

while not all(trouver(bloques,i)[0] == 1+max(trouver(bloques,x)[0] for _,x,xt in INSTS[i] if xt==0) for i in range(1,insts) if any(xt==0 for _,_,xt in INSTS[i])):
	for i in range(1, insts):
		if any(xt==0 for _,_,xt in INSTS[i]):
			_max = max(trouver(bloques,x)[0] for _,x,xt in INSTS[i] if xt==0)
			_avant_l, _avant_pos = trouver(bloques,i)
			bloques[_max+1] += [i]
			del bloques[_avant_l][_avant_pos]
		else:
			_max = 0
			_avant_l, _avant_pos = trouver(bloques,i)
			bloques[0] += [i]
			del bloques[_avant_l][_avant_pos]
	#	-----
	sleep(1)
	for l in bloques:
		print(l, [INSTS[i] for i in l])

while not [] in bloques:
	del bloques[bloques.index([])]

pos_inst = [trouver(bloques,i) for i in range(len(INSTS))]

for i in range(1, len(INSTS)):
	for X,x,xt in INSTS[i]:
		if xt == 0:
			pos_i_x, pos_i_y = pos_inst[x]
			pos_x_x, pos_x_y = pos_inst[i]

			x,y = pos_i_x, pos_i_y
			dx, dy = (pos_x_x-pos_i_x), (pos_x_y-pos_i_y)
			plt.arrow(x, y, dx, dy)
			#plt.plot([pos_i_x, pos_x_x], [pos_i_y, pos_x_y])

plt.show()