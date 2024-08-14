#pragma once

#include "insts.cuh"

//#define pseudo_rnd(x) ((x%7)*(x%11)*(x%27) % 121)*(x%77)
#define pseudo_rnd(x) ( (71*x + 313)%283 + (17*x + 223)%199 + (83*x + 983)%419 )

#define drop_vecteur__Xs 1
#define drop_vecteur__Ys 1
#define drop_vecteur__PARAMS 2 // Vecteur dropppable (qui peut etre = a 1), et %
#define drop_vecteur__Nom "Drop Vector"

uint drop_vecteur__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint drop_vecteur__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void drop_vecteur__init_poids(Inst_t * inst);

void drop_vecteur__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void drop_vecteur__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void drop_vecteur__pre_f(Inst_t * inst);

static fonctions_insts_t fi_drop_vecteur = {
	.Xs    =drop_vecteur__Xs,
	.PARAMS=drop_vecteur__PARAMS,
	.Nom   =drop_vecteur__Nom,
	//
	.calculer_P=drop_vecteur__calculer_P,
	.calculer_L=drop_vecteur__calculer_L,
	//
	.init_poids=drop_vecteur__init_poids,
	//
	.f =drop_vecteur__f,
	.df=drop_vecteur__df,
	//
	.pre_f=drop_vecteur__pre_f
};