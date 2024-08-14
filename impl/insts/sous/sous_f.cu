#include "sous.cuh"

__global__
static void kerd__sub1(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint * ts__d, uint mega_t,
	//
	float sng_soustraction)
{
	//
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	uint thy = threadIdx.y + blockIdx.y * blockDim.y;
	//
#define _y thx
#define _t thy
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		y[ty*Y + _y] = sng_soustraction*x0[tx0*X0 + _y];
	}
};

__global__
static void kerd__sub2(
	uint x0_t, uint X0, float * x0,
	uint x1_t, uint X1, float * x1,
	//
	uint    Y,
	float * y,
	//
	uint * ts__d, uint mega_t)
{
	//
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	uint thy = threadIdx.y + blockIdx.y * blockDim.y;
	//
#define _y thx
#define _t thy
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint tx1 = t_MODE(_t, mega_t-x1_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		y[ty*Y + _y] = x0[tx0*X0 + _y] - x1[tx1*X1 + _y];
	}
};

void sous__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0];	uint x0_t = inst->x_t[0];
	uint X1 = inst->x_Y[1];	uint x1_t = inst->x_t[1];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	bool x1_existe = (mega_t != 0 ? true : (x1_t != 1));
	//
	uint xs_existants = x0_existe + x1_existe;
	//
	if (xs_existants == 2) {
		kerd__sub2<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			x0_t, X0, x__d[0],
			x1_t, X1, x__d[1],
			//
			inst->Y,
			inst->y__d,
			//
			ts__d, mega_t
		);
	} else if (xs_existants == 1) {
		uint _i0 = (x0_existe ? 0 : 1);
		//
		float sng_soustraction = (_i0 == 0 ? +1 : -1);
		//
		kerd__sub1<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[_i0], inst->x_Y[_i0], x__d[_i0],
			//
			inst->Y,
			inst->y__d,
			//
			ts__d, mega_t,
			//
			sng_soustraction
		);
	} else if (xs_existants == 0) {
		inst_zero_mega_t(inst, mega_t);
	}
};