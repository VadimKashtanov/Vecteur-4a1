#include "canalisation.cuh"

__global__
static void d_kerd__canalisation(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y,
	float * y, float * dy,
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
		//float _x = x0[tx0*X0 + _y];
		float d_x = 0;
		FOR(0, c0, C0) {
			d_x += dy[ty*Y + c0*X0 + _y];
		};
		//
		atomicAdd(&dx0[tx0*X0 + _y], d_x);
	};
};

void canalisation__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint C0 = inst->params[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		d_kerd__canalisation<<<dim3(KERD(X0,16), KERD(GRAND_T,8)), dim3(16,8)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			mega_t,
			//
			C0
		);
	} else {
		//	rien
	}
};