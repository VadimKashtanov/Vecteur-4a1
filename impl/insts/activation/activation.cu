#include "activation.cuh"

uint activation__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint activation__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return Y;
};

void activation__init_poids(Inst_t * inst) {
	//inst->p__d;
};

void activation__pre_f(Inst_t * inst) {
	
};