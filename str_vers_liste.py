import struct as st

import matplotlib.pyplot as plt

with open('y_mdl.bin', 'rb') as co:
    bins = co.read()
    f = st.unpack('f'*int(len(bins)/4), bins)

plt.imshow([[f[y*54+x] for x in range(54)] for y in range(54)]); plt.show()
