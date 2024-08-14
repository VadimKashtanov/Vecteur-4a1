#include "activation.cuh"

__global__
static void kerd__activation(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y, uint    L,
	float * y, float * l,
	//
	uint mega_t,
	//
	uint activ)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float s = x0[tx0*X0 + _y];
		//
		float a = ACTIVATION(activ, s);
		//
		y[ty*Y + _y] = a;
		l[ty*L + _y] = s;
	};
};

void activation__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint x0_t = inst->x_t[0];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		kerd__activation<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y, inst->L,
			inst->y__d, inst->l__d,
			//
			mega_t,
			//
			inst->params[0]
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};