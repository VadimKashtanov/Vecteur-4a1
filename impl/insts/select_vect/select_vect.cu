#include "select_vect.cuh"

uint select_vect__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint select_vect__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void select_vect__init_poids(Inst_t * inst) {
	uint \
		Vect = inst->params[0],	\
		N    = inst->params[1];
	//
	ASSERT(inst->x_Y[0] % inst->Y == 0);
	ASSERT(inst->Y == Vect);
	ASSERT(Vect*N < inst->x_Y[0]);
};

void select_vect__pre_f(Inst_t * inst) {
	
};