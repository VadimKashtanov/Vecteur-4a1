#pragma once

#include "insts.cuh"

#define poid__Xs 0
#define poid__Ys 1
#define poid__PARAMS 0
#define poid__Nom "Poid"

uint poid__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint poid__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void poid__init_poids(Inst_t * inst);

void poid__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void poid__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void poid__pre_f(Inst_t * inst);

static fonctions_insts_t fi_poid = {
	.Xs    =poid__Xs,
	.PARAMS=poid__PARAMS,
	.Nom   =poid__Nom,
	//
	.calculer_P=poid__calculer_P,
	.calculer_L=poid__calculer_L,
	//
	.init_poids=poid__init_poids,
	//
	.f =poid__f,
	.df=poid__df,
	//
	.pre_f=poid__pre_f
};