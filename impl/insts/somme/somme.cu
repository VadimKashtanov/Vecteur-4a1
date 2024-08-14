#include "somme.cuh"

uint somme__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint somme__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void somme__init_poids(Inst_t * inst) {
	ASSERT(inst->Y == inst->x_Y[0] && inst->Y == inst->x_Y[1]);
};

void somme__pre_f(Inst_t * inst) {
	
};