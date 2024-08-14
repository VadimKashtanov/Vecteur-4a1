#pragma once

#include "insts.cuh"

#define isomme__Xs 1
#define isomme__Ys 1
#define isomme__PARAMS 1
#define isomme__Nom "I somme"

uint isomme__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint isomme__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void isomme__init_poids(Inst_t * inst);

void isomme__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void isomme__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void isomme__pre_f(Inst_t * inst);

static fonctions_insts_t fi_isomme = {
	.Xs    =isomme__Xs,
	.PARAMS=isomme__PARAMS,
	.Nom   =isomme__Nom,
	//
	.calculer_P=isomme__calculer_P,
	.calculer_L=isomme__calculer_L,
	//
	.init_poids=isomme__init_poids,
	//
	.f =isomme__f,
	.df=isomme__df,
	//
	.pre_f=isomme__pre_f
};