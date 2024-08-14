#include "opti.cuh"

__global__ static void kerd_sgd(
	uint t,
	float * p, float * dp,
	//
	float alpha,
	uint POIDS)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		float _grad = dp[thx];
		//
		float ch  = alpha * _grad;
		float reg = alpha * L2_regularisation * p[thx];
		//
		p[thx] -= (ch + reg);
	}
};

void sgd(
	Mdl_t   * mdl,
	float *** hist,
	uint         i,
	float    alpha,
	uint         t
) {
	FOR(0, i, mdl->insts) {
		Inst_t * inst = mdl->inst[i];
		//
		if (inst->P != 0) {
			kerd_sgd<<<dim3(KERD(mdl->inst[i]->P, 256)),dim3(256)>>>(
				t,
				inst->p__d, inst->dp__d,
				//
				alpha,
				inst->P
			);
		}
	}
	ATTENDRE_CUDA();
};