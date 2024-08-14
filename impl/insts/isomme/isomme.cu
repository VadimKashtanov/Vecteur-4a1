#include "isomme.cuh"

uint isomme__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint isomme__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void isomme__init_poids(Inst_t * inst) {
	uint C0 = inst->params[0];
	//
	ASSERT(inst->Y == C0);
	ASSERT(C0 > 0);
	//inst->p__d;
};

void isomme__pre_f(Inst_t * inst) {
	FOR(0, mega_t, MEGA_T) inst_zero_mega_t(inst, mega_t);
};