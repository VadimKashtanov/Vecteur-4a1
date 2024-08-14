#include "insts.cuh"

__global__
static void kerd_inst_zero_mega_t(float * y, uint Y, uint mega_t)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		y[t_MODE(_t, mega_t)*Y + _y] = 0.0;
	};
};

void inst_zero_mega_t(Inst_t * inst, uint mega_t) {
	//kerd_inst_zero_mega_t<<<DIM2(inst->Y, GRAND_T, 16,16)>>>(
	kerd_inst_zero_mega_t<<<dim3(KERD(inst->Y,16), KERD(GRAND_T,16)),dim3(16,16)>>>(
		inst->y__d,
		inst->Y,
		mega_t
	);
};