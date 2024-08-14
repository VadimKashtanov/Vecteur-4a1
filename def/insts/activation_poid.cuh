#pragma once

#include "insts.cuh"

#define activation_poid__Xs 1
#define activation_poid__Ys 1
#define activation_poid__PARAMS 1
#define activation_poid__Nom "Activ poid"

uint activation_poid__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint activation_poid__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void activation_poid__init_poids(Inst_t * inst);

void activation_poid__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void activation_poid__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void activation_poid__pre_f(Inst_t * inst);

static fonctions_insts_t fi_activation_poid = {
	.Xs    =activation_poid__Xs,
	.PARAMS=activation_poid__PARAMS,
	.Nom   =activation_poid__Nom,
	//
	.calculer_P=activation_poid__calculer_P,
	.calculer_L=activation_poid__calculer_L,
	//
	.init_poids=activation_poid__init_poids,
	//
	.f =activation_poid__f,
	.df=activation_poid__df,
	//
	.pre_f=activation_poid__pre_f
};