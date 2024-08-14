#include "opti.cuh"

#define rms 0.9

__global__ static void kerd_rmsprop(
	uint t,
	float * p, float * dp,
	float * s,
	float alpha,
	uint POIDS)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		float _grad = dp[thx];
		//
		float _s = rms*s[thx] + (1-rms)*_grad*_grad;
		//
		s[thx] = _s;
		//
		float ch  = alpha * _grad / (sqrtf(_s + 1e-8));
		float reg = alpha * L2_regularisation * p[thx];
		//
		p[thx] -= (ch + reg);
	}
};

void rmsprop(
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
			kerd_rmsprop<<<dim3(KERD(mdl->inst[i]->P, 256)),dim3(256)>>>(
				t,
				inst->p__d, inst->dp__d,
				hist[0][i],
				alpha,
				inst->P
			);
		}
	}
	ATTENDRE_CUDA();
};