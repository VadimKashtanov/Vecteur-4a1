import struct as st

def ecrire_str(s):
	return st.pack('I', len(s)) + s.encode()

def lire_str(bins):
	_len, = st.unpack('I', bins[:4])
	return b''.join(st.unpack('c'*_len, bins[4:4+_len])).decode(), bins[4+_len:]

def st_lire(bins, taille):
	I = st.calcsize(taille)
	return list(st.unpack(taille, bytes(bins[:I]))), bins[I:]