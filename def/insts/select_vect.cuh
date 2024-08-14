#pragma once

#include "insts.cuh"

#define select_vect__Xs 1
#define select_vect__PARAMS 2 // Vect_Taille, N-eme
#define select_vect__Nom "Select N-eme vect"

uint select_vect__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint select_vect__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void select_vect__init_poids(Inst_t * inst);

void select_vect__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void select_vect__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void select_vect__pre_f(Inst_t * inst);

static fonctions_insts_t fi_select_vect = {
	.Xs    =select_vect__Xs,
	.PARAMS=select_vect__PARAMS,
	.Nom   =select_vect__Nom,
	//
	.calculer_P=select_vect__calculer_P,
	.calculer_L=select_vect__calculer_L,
	//
	.init_poids=select_vect__init_poids,
	//
	.f =select_vect__f,
	.df=select_vect__df,
	//
	.pre_f=select_vect__pre_f
};