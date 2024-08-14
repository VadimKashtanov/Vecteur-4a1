#include "softmax.cuh"

uint softmax__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint softmax__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void softmax__init_poids(Inst_t * inst) {
	ASSERT(inst->Y == inst->x_Y[0]);
	ASSERT(inst->Y % inst->params[0] == 0);
	//inst->p__d;
};

void softmax__pre_f(Inst_t * inst) {
	
};