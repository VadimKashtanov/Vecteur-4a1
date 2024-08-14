#include "positionnal.cuh"

__global__
static void kerd__positionnal(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint L, uint N)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float e = x0[tx0*X0 + _y];
		//
		uint i = (_y % N);
		uint k = (i%2==0 ? i/1 : (i-1)/2);
		float a;
		if (i % 2 == 0) a = sin((float)i / powf(10000.0, 2*k/(float)L));
		if (i % 2 == 1) a = cos((float)i / powf(10000.0, 2*k/(float)L));
		//
		y[ty*Y + _y] = e+a;
	};
};

void positionnal__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint x0_t = inst->x_t[0];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		kerd__positionnal<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y,
			inst->y__d,
			//
			mega_t,
			//
			inst->params[0], inst->params[1]
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};