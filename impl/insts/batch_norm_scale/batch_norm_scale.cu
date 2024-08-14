#include "batch_norm_scale.cuh"

uint batch_norm_scale__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	uint \
		C0 = params[0];
	return 2 * C0;
};

uint batch_norm_scale__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

void batch_norm_scale__init_poids(Inst_t * inst) {
	uint \
		C0 = inst->params[0];
	//
	ASSERT(inst->Y == inst->x_Y[0]);
	//
	float p[inst->P];
	FOR(0, i, inst->P) p[i] = poid_1_1();

	CONTROLE_CUDA(cudaMemcpy(inst->p__d, p, sizeof(float)*inst->P, cudaMemcpyHostToDevice));
};

void batch_norm_scale__pre_f(Inst_t * inst) {
	
};