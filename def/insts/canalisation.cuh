#pragma once

#include "insts.cuh"

#define canalisation__Xs 1
#define canalisation__Ys 1
#define canalisation__PARAMS 1
#define canalisation__Nom "Canalisation"

uint canalisation__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint canalisation__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void canalisation__init_poids(Inst_t * inst);

void canalisation__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void canalisation__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void canalisation__pre_f(Inst_t * inst);

static fonctions_insts_t fi_canalisation = {
	.Xs    =canalisation__Xs,
	.PARAMS=canalisation__PARAMS,
	.Nom   =canalisation__Nom,
	//
	.calculer_P=canalisation__calculer_P,
	.calculer_L=canalisation__calculer_L,
	//
	.init_poids=canalisation__init_poids,
	//
	.f =canalisation__f,
	.df=canalisation__df,
	//
	.pre_f=canalisation__pre_f
};