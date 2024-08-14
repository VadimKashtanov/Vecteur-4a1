#pragma once

// miu = sum(e) / len(l)

#include "insts.cuh"

#define batch_miu__Xs 1
#define union_miu__Ys 1
#define batch_miu__PARAMS 1 //C0
#define batch_miu__Nom "Batch Miu"

uint batch_miu__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint batch_miu__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void batch_miu__init_poids(Inst_t * inst);

void batch_miu__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void batch_miu__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void batch_miu__pre_f(Inst_t * inst);

static fonctions_insts_t fi_batch_miu = {
	.Xs    =batch_miu__Xs,
	.PARAMS=batch_miu__PARAMS,
	.Nom   =batch_miu__Nom,
	//
	.calculer_P=batch_miu__calculer_P,
	.calculer_L=batch_miu__calculer_L,
	//
	.init_poids=batch_miu__init_poids,
	//
	.f =batch_miu__f,
	.df=batch_miu__df,
	//
	.pre_f=batch_miu__pre_f
};