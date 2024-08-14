#include "canalisation.cuh"

__global__
static void kerd__canalisation(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint C0)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float _x = x0[tx0*X0 + _y];
		FOR(0, c0, C0) {
			y[ty*Y + c0*X0 + _y] = _x;
		}
	};
};

void canalisation__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint C0 = inst->params[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		kerd__canalisation<<<dim3(KERD(X0,16), KERD(GRAND_T,8)), dim3(16,8)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
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