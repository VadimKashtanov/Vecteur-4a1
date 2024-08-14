#include "isomme.cuh"

static __global__ void kerd_isomme(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
	//
	uint mega_t,
	//
	uint C0)
{
	uint _x = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_x < X0 && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float s = x0[tx0*X0 + _x];
		//
		uint c0 = (  _x - (_x%(X0/C0))  )/(X0/C0);
		//printf("%i\n", c0);
		//
		atomicAdd(&y[ty*Y + c0], s);
	};
}

void isomme__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint \
		C0 = inst->params[0];
	//
	uint x0_t = inst->x_t[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	ASSERT(x0_existe);
	//
	if (x0_existe) {
		kerd_isomme<<<dim3(KERD(inst->x_Y[0],16), KERD(GRAND_T,16)), dim3(16,16)>>>(
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
		//inst_zero_mega_t(inst, mega_t);
	}
};