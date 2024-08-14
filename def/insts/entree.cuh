#pragma once

#include "insts.cuh"

#define entree__Xs 1
#define entree__Ys INTERVS
#define entree__PARAMS 0
#define entree__Nom "Entree"

uint entree__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint entree__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void entree__init_poids(Inst_t * inst);

void entree__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void entree__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void entree__pre_f(Inst_t * inst);

static fonctions_insts_t fi_entree = {
	.Xs    =entree__Xs,
	.PARAMS=entree__PARAMS,
	.Nom   =entree__Nom,
	//
	.calculer_P=entree__calculer_P,
	.calculer_L=entree__calculer_L,
	//
	.init_poids=entree__init_poids,
	//
	.f =entree__f,
	.df=entree__df,
	//
	.pre_f=entree__pre_f
};