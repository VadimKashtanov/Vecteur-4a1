#include "select_vect.cuh"

__global__
static void kerd__select_vect(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint Vect, uint N)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float s = x0[tx0*X0 + N*Vect + _y];
		//
		y[ty*Y + _y] = s;
	};
};

void select_vect__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint \
		Vect = inst->params[0],	\
		N    = inst->params[1];
	//
	uint x0_t = inst->x_t[0];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		kerd__select_vect<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y,
			inst->y__d,
			//
			mega_t,
			//
			Vect, N
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};