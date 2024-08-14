#include "activation_poid.cuh"

uint activation_poid__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return Y;
};

uint activation_poid__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return Y;
};

void activation_poid__init_poids(Inst_t * inst) {
	float p[inst->P];
	FOR(0, i, inst->P) p[i] = poid_1_1() * 0 * 0.5;

	CONTROLE_CUDA(cudaMemcpy(inst->p__d, p, sizeof(float)*inst->P, cudaMemcpyHostToDevice));
};

void activation_poid__pre_f(Inst_t * inst) {
	
};