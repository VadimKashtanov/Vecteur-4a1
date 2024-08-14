#! /usr/bin/python3

N = 32#64#32
INTERVALLE_MAX = 64
DEPART = INTERVALLE_MAX * N

INTERVS = 1, 4, 16, 64

import struct as st

with open("structure_generale.bin", 'rb') as co:
	bins = co.read()
	(I,) = st.unpack('I', bins[:4])
	elements = st.unpack('I'*int(len(bins[4:])/4), bins[4:])
	#
	MEGA_T, = elements