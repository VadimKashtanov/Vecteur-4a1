#include "batch_miu.cuh"

uint batch_miu__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint batch_miu__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void batch_miu__init_poids(Inst_t * inst) {
	
};

void batch_miu__pre_f(Inst_t * inst) {
	FOR(0, mega_t, MEGA_T) inst_zero_mega_t(inst, mega_t);
};