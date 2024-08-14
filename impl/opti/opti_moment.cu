#include "opti.cuh"

#define moment_p 0.9

__global__ static void kerd_moment(
	uint t,
	float * p, float * dp,
	float * m,
	float alpha,
	uint POIDS)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		float _grad = dp[thx];
		//
		float _m = (1-moment_p)*m[thx] + moment_p*_grad;
		m[thx] = _m;
		//
		float ch  = alpha * _m;
		float reg = alpha * L2_regularisation * p[thx];
		//
		p[thx] -= (ch + reg);
	}
};

void moment(
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
			kerd_moment<<<dim3(KERD(mdl->inst[i]->P, 256)),dim3(256)>>>(
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