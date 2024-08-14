class Dico:
	i = None
	#
	X  = []
	x  = []
	xt = []
	#
	y = 0
	#
	p = []
	#
	sortie = False
	# 
	def __init__(self, **karg):
		for k,v in karg.items():
			self[k] = v

	def __setitem__(self, k, v):
		if k in ('i', 'X', 'x', 'xt', 'y', 'p', 'sortie'):
			self.__setattr__(f'{k}', v)
		else:
			raise Exception(f"Argument {k} n'existe pas")

	def __getitem__(self, k):
		if k in ('i', 'X', 'x', 'xt', 'y', 'p', 'sortie'):
			return self.__getattribute__(f'{k}')

	def __str__(self):
		montrer = lambda obj: (obj if type(obj) in (int, type(None)) else f'<id={id(obj)}>')
		
		dans_str = lambda s, taille: (s+" "*(taille-len(s)) if len(s)<taille else s)

		i = dans_str(str(self.i ).replace("tkinter_cree_dossier.tkinter_insts.", ""), 30)
		X = dans_str(str(self.X ), 20)
		x = dans_str(str(list(map(montrer,self.x))), 15)
		xt= dans_str(str(self.xt), 10)
		y = dans_str(str(self.y ), 5)
		p = dans_str(str(self.p ), 25)

		return f'<id={id(self)}>   i:{i},  X:{X}, x:{x}, xt:{xt}, y:{y}, p:{p}, sortie:{self.sortie}'


	def copier(self):
		return Dico(
			i=self.i,
			X=self.X,
			x=self.x,
			xt=self.xt,
			y=self.y,
			p=self.p,
			sortie=self.sortie
		)