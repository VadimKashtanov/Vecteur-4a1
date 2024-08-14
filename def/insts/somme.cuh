#pragma once

#include "insts.cuh"

#define somme__Xs 2
#define somme__Ys 1
#define somme__PARAMS 0
#define somme__Nom "Somme"

uint somme__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint somme__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void somme__init_poids(Inst_t * inst);

void somme__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void somme__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void somme__pre_f(Inst_t * inst);

static fonctions_insts_t fi_somme = {
	.Xs    =somme__Xs,
	.PARAMS=somme__PARAMS,
	.Nom   =somme__Nom,
	//
	.calculer_P=somme__calculer_P,
	.calculer_L=somme__calculer_L,
	//
	.init_poids=somme__init_poids,
	//
	.f =somme__f,
	.df=somme__df,
	//
	.pre_f=somme__pre_f
};