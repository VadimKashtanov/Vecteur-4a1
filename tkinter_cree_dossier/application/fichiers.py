from tkinter_cree_dossier.application._etc import *

from tkinter_cree_dossier.tkinter_insts import *
from tkinter_cree_dossier.tkinter_modules_inst_liste import *

from etc import *

class Fichiers(tk.LabelFrame):
	def __init__(self, parent, application, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, text='Fichiers', *args, **kwargs)
		self.application = application

		img_ouvrire     = tk.PhotoImage(file="tkinter_cree_dossier/img/ouvrire.png"    )
		img_enregistrer = tk.PhotoImage(file="tkinter_cree_dossier/img/enregistrer.png")
		img_bin         = tk.PhotoImage(file="tkinter_cree_dossier/img/bin.png"        )

		# Create arrow buttons
		btn_enregistrer = tk.Button(self, image=img_enregistrer, command=self.cmd_enregistrer)
		btn_enregistrer.grid(row=0, column=0, sticky='nsew')
		#
		btn_ouvrire = tk.Button(self, image=img_ouvrire,     command=self.cmd_ouvrire)
		btn_ouvrire.grid(row=0, column=1, sticky='nsew')
		#
		btn_bin = tk.Button(self, image=img_bin,         command=self.cmd_bin)
		btn_bin.grid(row=0, column=2, sticky='nsew')

		self.gc_img = [img_ouvrire, img_enregistrer, img_bin]
		self.gc_btn = [btn_ouvrire, btn_enregistrer, btn_bin]

	def cmd_ouvrire(self):
		self.application.x, self.application.y = 0, 0
		#
		fichier = filedialog.askopenfilename(filetypes = (('module', '*.module'), ('Tous les fichier', '*.*')))
		#
		with open(fichier, 'rb') as co:
			bins = (co.read())
			#
			(L_f, L_c,), bins = st_lire(bins, 2*'I')
			#
			self.application.canvas.tout_supprimer()
			#
			for f in range(L_f):
				(ID,),bins = st_lire(bins, 1*'I')
				(x,y),bins = st_lire(bins, 2*'I')
				self.application.add_frame(modules_models[ID](), x=x, y=y)
				
				#	==== X ====
				for _x_ in range(len(self.application.frames[-1].module.X)):
					s, bins = lire_str(bins)
					self.application.frames[-1].xs[_x_].val.set(s)

				#	==== P ====
				for _p_ in range(len(self.application.frames[-1].module.params)):
					s, bins = lire_str(bins)
					self.application.frames[-1].ps[_p_].val.set(s)

				#	==== Y ====
				for _y_ in range(len(self.application.frames[-1].module.Y)):
					s, bins = lire_str(bins)
					self.application.frames[-1].ys[_y_].val.set(s)

				#
				self.application.frames[-1].mettre_a_jour_module()
			#
			for c in range(L_c):
				(iA,sA, iB,eB, t), bins = st_lire(bins, 5*'I')
				self.application.canvas.connections += [[(iA,sA),(iB,eB),-t]]
		#	--
		self.application.canvas.update()
		self.application.canvas.update_lines()

	def cmd_enregistrer(self):
		self.application.canvas.recentrer()
		self.application.re_ordonner_frames()
		#
		fichier = filedialog.asksaveasfilename(filetypes = (('module', '*.module'), ('Tous les fichier', '*.*')))
		#
		with open(fichier, 'wb') as co:
			co.write(st.pack('II', len(self.application.frames), len(self.application.canvas.connections)))
			#
			for f in self.application.frames:
				f.mettre_a_jour_module()
				#
				co.write(st.pack('III', modules_models.index(type(f.module)), f.winfo_x(), f.winfo_y()))
				#
				for _x_ in range(len(f.module.X     )): co.write(ecrire_str(f.xs[_x_].val.get()))
				for _x_ in range(len(f.module.params)): co.write(ecrire_str(f.ps[_x_].val.get()))
				for _x_ in range(len(f.module.Y     )): co.write(ecrire_str(f.ys[_x_].val.get()))

			for (iA,sA),(iB,eB),t in self.application.canvas.connections:
				co.write(st.pack('IIIII', iA,sA,iB,eB,abs(t)))

	def cmd_bin(self):
		self.application.re_ordonner_frames()
		
		#	=======================================================================
		#	Etape 1 : Union & indéxation
		#	Assemblage de toutes les instructions dans un seul ix. Puis indexation des entrés non `None`
		#	=======================================================================

		ix = []
		depart_i = []
		depart = 0
		for f in self.application.frames:
			f.mettre_a_jour_module()
			#
			m = f.module
			try: m.cree_ix()
			except Exception as e: raise Exception(f"Erreur dans frame={f.numero}, module : {m}")
			#
			for i in range(len(m.ix)):
				assert all(m.ix[i]['x'][j] in m.ix or m.ix[i]['x'][j] == None for j in range(len(m.ix[i]['x'])))
			#
			for _l in m.ix:
				ix += [_l]
				#
				_l['x'] = [(depart+m.ix.index(x) if x!=None else None) for x in _l['x']]
			#
			depart_i += [depart]
			depart += len(m.ix)

		#	Voyons la liste des sorties de chaque frame
		#sorties = [i for i in range(len(ix)) if ix[i]['sortie']]
		
		#	=======================================================================
		#	Touts les `None` sont des entrés ou sorties
		#	Etape 2 : Connection des entrés et sorties possibles avec self.canvas.connections (x et xt)
		#	=======================================================================

		les_None = [(i,x) for i in range(len(ix)) for x in range(len(ix[i]['x'])) if ix[i]['x'][x] == None]
		#
		les_None_par_inst = [[] for i in range(len(self.application.frames))]
		for inst in range(len(self.application.frames)):
			for (i,x) in les_None:
				if i >= depart_i[inst]:
					les_None_par_inst[inst] += [(i,x)]
		#
		for (iA,sA),(iB,eB),t in self.application.canvas.connections:
			i, x = les_None_par_inst[iB][eB]
			#
			len_ix    = len(self.application.frames[iA].module.ix)
			#print([depart_i[iA]+j for j in range(len_ix) if ix[depart_i[iA]+j]['sortie']], sA)
			sorties_A = [depart_i[iA]+j for j in range(len_ix) if ix[depart_i[iA]+j]['sortie']][sA]
			#
			ix[i]['x' ][x] = sorties_A
			ix[i]['xt'][x] = t

		#	=======================================================================
		#					Etape 3 : Supprimer les MODULE_i_Y
		#	=======================================================================

		while [i['i'] for i in ix].count(i_Y) != 0:
			module_i_y = [i['i'] for i in ix].index(i_Y)

			i_y_pointe_vers   = ix[module_i_y]['x' ][0]
			i_y_pointe_vers_t = ix[module_i_y]['xt'][0]

			#	1 : Rediriger les x et xt des MODULE_i_Y vers les bonnes

			for i in ix:
				for arg,pos in enumerate(i['x']):
					if pos == module_i_y:
						i['x' ][arg] = i_y_pointe_vers
						i['xt'][arg] = i_y_pointe_vers_t

			#	2 : Supprimer MODULE_i_Y
			del ix[module_i_y]

			for i in ix:
				for arg,pos in enumerate(i['x']):
					if pos != None:
						if pos > module_i_y:
							i['x'][arg] -= 1

		#	=======================================================================
		#	Etape 4 : Identifier les None Restant comme entrées
		#	=======================================================================

		entrées   = 0
		la_sortie = len(ix)-1

		assert la_sortie == len(ix)-1

		#	=======================================================================
		#	Etape Finale : Ecrire .bin
		#	=======================================================================

		print(f'La sortie = {la_sortie}')
		print(f"L'entrée  = {entrées}")

		#	fichier.st.bin
		bins = b''
		bins += st.pack('I', len(ix))
		for pos,i in enumerate(ix):
			#	Verification
			inst = i['i'](i['X'], i['y'], i['p'])
			print('{pos:3}| '.format(pos=pos, i=i) + str(i))
			inst.assert_coherance()
			if pos != 0:
				for j,_x in enumerate(i['x']):
					assert ix[_x]['y'] == i['X'][j]
			#	ID
			bins += st.pack('I', liste_insts.index(inst.__class__))
			#	X
			for _x in range(len(i['x'])):
				est_une_entree = (i['x'][_x] == None)
				#bins += st.pack('I', int(est_une_entree))
				#
				if not est_une_entree:
					X, x, xt = i['X'][_x], i['x'][_x], abs(i['xt'][_x])
				else:
					X, x, xt = i['y'], (1 << 32)-1, 0
				#
				bins += st.pack('III', X, x, xt)
			#	Y
			bins += st.pack('I', i['y'])
			#	Params
			bins += st.pack('I'*len(i['p']), *i['p'])
		#
		bins += st.pack('I', entrées  )
		bins += st.pack('I', la_sortie)

		#	=======================================================================
		#	--- IO ---
		#	=======================================================================

		fichier = filedialog.asksaveasfilename(filetypes = (('pre_mdl', '*.st.bin'), ('Tous les fichier', '*.*')))
		with open(fichier, 'wb') as co:
			co.write(bins)