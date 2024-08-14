#include "normalisation.cuh"

uint normalisation__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint normalisation__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0; //max, min
};

void normalisation__init_poids(Inst_t * inst) {
	uint C0     = inst->params[0];
	//
	ASSERT(inst->Y == inst->x_Y[0]);
	ASSERT(C0 > 0);
};

void normalisation__pre_f(Inst_t * inst) {
	
};