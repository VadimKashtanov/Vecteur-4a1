#include "batch_variance.cuh"

uint batch_variance__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint batch_variance__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void batch_variance__init_poids(Inst_t * inst) {
};

void batch_variance__pre_f(Inst_t * inst) {
	FOR(0, mega_t, MEGA_T) inst_zero_mega_t(inst, mega_t);
};