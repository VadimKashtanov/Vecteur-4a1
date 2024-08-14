#pragma once

#include "insts.cuh"

#define imaxmin__Xs 1
#define imaxmin__Ys 1
#define imaxmin__PARAMS 1
#define imaxmin__Nom "I max/min"

uint imaxmin__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint imaxmin__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void imaxmin__init_poids(Inst_t * inst);

void imaxmin__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void imaxmin__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void imaxmin__pre_f(Inst_t * inst);

static fonctions_insts_t fi_imaxmin = {
	.Xs    =imaxmin__Xs,
	.PARAMS=imaxmin__PARAMS,
	.Nom   =imaxmin__Nom,
	//
	.calculer_P=imaxmin__calculer_P,
	.calculer_L=imaxmin__calculer_L,
	//
	.init_poids=imaxmin__init_poids,
	//
	.f =imaxmin__f,
	.df=imaxmin__df,
	//
	.pre_f=imaxmin__pre_f
};