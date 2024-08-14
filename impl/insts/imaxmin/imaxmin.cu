#include "imaxmin.cuh"

#include "../impl_template/tmpl_etc.cu"

uint imaxmin__calculer_P(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	return 0;
};

uint imaxmin__calculer_L(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]) {
	uint C0     = params[0];
	return 0;//2 * C0; //max, min
};

void imaxmin__init_poids(Inst_t * inst) {
	uint C0     = inst->params[0];
	//
	ASSERT(inst->Y == 2*C0);
	ASSERT(C0 > 0);
};

static __global__ void poser_FLT_MAX(
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
		uint ty  = t_MODE(_t, mega_t     );
		//
		y[ty*Y + _y] = (_y%2==0 ? -FLT_MAX : FLT_MAX);	//max , min
	}
}

void imaxmin__pre_f(Inst_t * inst) {
	uint C0 = inst->params[0];
	FOR(0, mega_t, MEGA_T) {
		poser_FLT_MAX<<<dim3(KERD(inst->Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->Y,
			inst->y__d,
			//
			mega_t,
			//
			C0
		);
	}
};