#pragma once

#include "insts.cuh"

#define matmul_poid_AP__Xs 1
#define matmul_poid_AP__Ys 1
#define matmul_poid_AP__PARAMS 4
#define matmul_poid_AP__Nom "Mat mul poid"

uint matmul_poid_AP__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint matmul_poid_AP__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void matmul_poid_AP__init_poids(Inst_t * inst);

void matmul_poid_AP__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void matmul_poid_AP__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void matmul_poid_AP__pre_f(Inst_t * inst);

static fonctions_insts_t fi_matmul_poid_AP = {
	.Xs    =matmul_poid_AP__Xs,
	.PARAMS=matmul_poid_AP__PARAMS,
	.Nom   =matmul_poid_AP__Nom,
	//
	.calculer_P=matmul_poid_AP__calculer_P,
	.calculer_L=matmul_poid_AP__calculer_L,
	//
	.init_poids=matmul_poid_AP__init_poids,
	//
	.f =matmul_poid_AP__f,
	.df=matmul_poid_AP__df,
	//
	.pre_f=matmul_poid_AP__pre_f
};