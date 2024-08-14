#include "batch_variance.cuh"

static __global__ void kerd_batch_variance(
	uint x0_t, uint X0, float * x0,
	uint x1_t, uint X1, float * x1,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint C0)
{
	uint _x = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_x < X0 && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		//uint tx1 = t_MODE(_t, mega_t-x1_t);
		//uint ty  = t_MODE(_t, mega_t     );
		//
		float s = x0[tx0*X0 + _x];
		//
		uint c0 = (_x - (_x%(X0/C0))  )/(X0/C0);
		//
		float miu = x1[0*X1 + c0];
		//
		uint LEN_L = X0 / C0;
		atomicAdd(&y[0*Y + c0], powf(s - miu, 2) / (float)(LEN_L*GRAND_T));
	};
}

void batch_variance__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint \
		C0 = inst->params[0];
	//
	uint x0_t = inst->x_t[0];
	uint x1_t = inst->x_t[1];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	bool x1_existe = (mega_t != 0 ? true : (x1_t != 1));
	//
	ASSERT(x0_existe && x1_existe);
	//
	if (x0_existe && x1_existe) {
		kerd_batch_variance<<<dim3(KERD(inst->x_Y[0],16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			inst->x_t[1], inst->x_Y[1], x__d[1],
			//
			inst->Y,
			inst->y__d,
			//
			mega_t,
			//
			C0
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};