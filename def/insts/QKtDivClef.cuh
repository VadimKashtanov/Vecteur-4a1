// (Q @ K.t) / Clef**.5

#pragma once

#include "insts.cuh"

#define QKtDivClef__Xs 2
#define QKtDivClef__Ys 1
#define QKtDivClef__PARAMS 4
#define QKtDivClef__Nom "(Q @ K.t)/Clef**.5"

uint QKtDivClef__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
uint QKtDivClef__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);

void QKtDivClef__init_poids(Inst_t * inst);

void QKtDivClef__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t);
void QKtDivClef__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);

void QKtDivClef__pre_f(Inst_t * inst);

static fonctions_insts_t fi_QKtDivClef = {
	.Xs    =QKtDivClef__Xs,
	.PARAMS=QKtDivClef__PARAMS,
	.Nom   =QKtDivClef__Nom,
	//
	.calculer_P=QKtDivClef__calculer_P,
	.calculer_L=QKtDivClef__calculer_L,
	//
	.init_poids=QKtDivClef__init_poids,
	//
	.f =QKtDivClef__f,
	.df=QKtDivClef__df,
	//
	.pre_f=QKtDivClef__pre_f
};