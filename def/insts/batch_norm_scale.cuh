#pragma once

// bn  = (e - miu) / sqrt(var**2 + 1e-9)
// bn  = alpha * bn + beta

#include "insts.cuh"

#define batch_norm_scale__Xs 3
#define batch_norm_scale__Ys 1
#define batch_norm_scale__PARAMS 1 // C0
#define batch_norm_scale__Nom "Batch Norm Scale"

uint batch_norm_scale__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint batch_norm_scale__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void batch_norm_scale__init_poids(Inst_t * inst);

void batch_norm_scale__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void batch_norm_scale__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void batch_norm_scale__pre_f(Inst_t * inst);

static fonctions_insts_t fi_batch_norm_scale = {
	.Xs    =batch_norm_scale__Xs,
	.PARAMS=batch_norm_scale__PARAMS,
	.Nom   =batch_norm_scale__Nom,
	//
	.calculer_P=batch_norm_scale__calculer_P,
	.calculer_L=batch_norm_scale__calculer_L,
	//
	.init_poids=batch_norm_scale__init_poids,
	//
	.f =batch_norm_scale__f,
	.df=batch_norm_scale__df,
	//
	.pre_f=batch_norm_scale__pre_f
};