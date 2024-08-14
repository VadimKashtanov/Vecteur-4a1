#include "poid.cuh"

__global__
static void d_kerd__poid(
	float * p, float * dp,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint mega_t)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	uint thy = threadIdx.y + blockIdx.y * blockDim.y;
	//
#define _y thx
#define _t thy
	//
	if (_y < Y && _t < GRAND_T) {
		uint ty  = t_MODE(_t, mega_t);
		
		//y[ty*Y + _y] = p[_y];
		atomicAdd(&dp[_y], dy[ty*Y + _y]);
	};
};

void poid__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint Y = inst->Y;
	d_kerd__poid<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
		inst->p__d, inst->dp__d,
		//
		inst->Y,
		inst->y__d, inst->dy__d,
		//
		mega_t
	);
};