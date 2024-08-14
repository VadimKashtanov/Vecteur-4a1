#include "concatenation.cuh"

uint concatenation__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint concatenation__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void concatenation__init_poids(Inst_t * inst) {
	ASSERT(inst->Y == inst->x_Y[0]);
	//inst->p__d;
};

void concatenation__pre_f(Inst_t * inst) {
	
};