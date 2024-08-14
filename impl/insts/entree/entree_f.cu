#include "entree.cuh"

__global__
static void kerd__entree(
	uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint * ts__d, uint mega_t)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	uint thy = threadIdx.y + blockIdx.y * blockDim.y;
	//
#define _y thx
#define _t thy
	//
	if (_y < Y && _t < GRAND_T) {
		uint ts = ts__d[_t] + mega_t;
		uint ty = t_MODE(_t, mega_t);
		//
		y[ty*Y + _y] = x0[ts*X0 + _y];
	};
};

void entree__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint Y = inst->Y;
	//
	kerd__entree<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
		inst->x_Y[0], x__d[0],
		//
		inst->Y,
		inst->y__d,
		//
		ts__d, mega_t
	);
};