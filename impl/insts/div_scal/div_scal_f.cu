#include "div_scal.cuh"

static __global__ void kerd_div_scal(
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
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint tx1 = t_MODE(_t, mega_t-x1_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		uint c0 = (  _y - (_y%(X0/C0))  )/(X0/C0);
		//
		y[ty*Y + _y] = x0[tx0*X0 + _y] / x1[tx1*X1 + c0];
		//printf("%f\n", x1[tx1*X1 + C0]);
	};
}

void div_scal__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint \
		C0 = inst->params[0];
	//
	uint x0_t = inst->x_t[0];
	uint x1_t = inst->x_t[1];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	bool x1_existe = (mega_t != 0 ? true : (x1_t != 1));
	//
	ASSERT(x0_existe && x1_existe);
	//
	//inst_zero_mega_t(inst, mega_t);
	//
	if (x0_existe && x1_existe) {
		kerd_div_scal<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
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