#include "batch_miu.cuh"

static __global__ void kerd_batch_miu(
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
		uint LEN_L = X0 / C0;
		//
		uint c0 = ( _x - (_x % LEN_L) )/LEN_L;
		assert(c0 < C0);
		//
		//if (s != s) {
			//printf("tx0=%i _x=%i X0=%i c0=%i C0=%i %f %f %f\n", tx0, _x, X0, c0, C0, x0[tx0*X0 + _x-1], s, x0[tx0*X0 + _x+1]);
			//assert(0);
		//}
		//float somme = 0;
		/*FOR(0, _t, GRAND_T) {
			uint tx0 = t_MODE(_t, mega_t-x0_t);
			somme += x0[tx0*X0 + _x];
			if (x0[tx0*X0 + _x] != x0[tx0*X0 + _x]) printf("%i %i %f\n", tx0, _x, x0[tx0*X0 + _x]);
			//assert(x0[tx0*X0 + _x] != x0[tx0*X0 + _x]);
		}*/
		//y[0*Y + c0] = somme / (float)(LEN_L * GRAND_T);
		atomicAdd(&y[0*Y + c0], s / (float)(LEN_L * GRAND_T));
	};
}

void batch_miu__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
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
		kerd_batch_miu<<<dim3(KERD(inst->x_Y[0],16), KERD(GRAND_T,16)), dim3(16,16)>>>(
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