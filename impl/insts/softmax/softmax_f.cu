#include "softmax.cuh"

__global__
static void kerd__softmax(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint Vect)
{
	uint _v = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	uint Vects = X0 / Vect;
	//
	if (_v < Vects && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float somme = 0;
		FOR(0, i, Vect) {
			somme += expf(x0[tx0*X0 + _v*Vect + i]);
		}
		FOR(0, i, Vect) {
			float val = expf(x0[tx0*X0 + _v*Vect + i]) / somme;
			assert(val == val);	//	Softmax
			y[ty*X0 + _v*Vect + i] = val;
		}
	};
};

void softmax__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint \
		Vect = inst->params[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	uint Vects = X0 / Vect;
	//
	if (x0_existe) {
		kerd__softmax<<<dim3(KERD(Vects,16), KERD(GRAND_T,8)), dim3(16,8)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y,
			inst->y__d,
			//
			mega_t,
			//
			Vect
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};