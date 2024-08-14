#include "transpose2d.cuh"

uint transpose2d__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint transpose2d__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void transpose2d__init_poids(Inst_t * inst) {
	uint * params = inst->params;
	uint \
		Ax =params[0],	\
		Ay =params[1],	\
		C0 =params[2];
	//
	ASSERT(inst->Y      == C0 * Ay*Ax);
	ASSERT(inst->x_Y[0] == C0 * Ax*Ay);
};

void transpose2d__pre_f(Inst_t * inst) {
	
};