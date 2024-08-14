#include "imaxmin.cuh"

#include "../impl_template/tmpl_etc.cu"

/*static __global__ void d_kerd_imaxmin(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y, uint L,
	float * y, float * l,
	float * dy,
	//
	uint mega_t,
	//
	uint C0)
{
	//uint _x = threadIdx.x + blockIdx.x * blockDim.x;
	uint thx = threadIdx.x;
	uint _c0 = blockIdx.x;
	//
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		uint _X = X0 / C0;
		//
		float dmax = dy[ty*2*C0 + 2*_c0 + 0];
		float dmin = dy[ty*2*C0 + 2*_c0 + 1];
		uint omax = (uint)l[ty*2*C0 + 2*_c0 + 0];
		uint omin = (uint)l[ty*2*C0 + 2*_c0 + 1];
		//
		atomicAdd(&dx0[tx0*X0 + _c0*_X + omax], dmax);
		atomicAdd(&dx0[tx0*X0 + _c0*_X + omin], dmin);
	};
}*/

static __global__ void d_kerd_imaxmin(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y, //uint    L,
	float * y, //float * l,
	float * dy,
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
		uint _X  = X0 / C0;
		uint _c0 = (_x - (_x%_X)) / _X;
		//
		float __x = x0[tx0*X0 + _x ];
		if (y[ty*Y + _c0*2+0] == __x) {	//max
			atomicAdd(&dx0[tx0*X0 + _x ], dy[ty*Y + _c0*2+0]);
		}
		if (y[ty*Y + _c0*2+1] == __x) {	//min
			atomicAdd(&dx0[tx0*X0 + _x ], dy[ty*Y + _c0*2+1]);
		}
	}
}

void imaxmin__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint \
		C0     = inst->params[0];
	//
	uint x0_t = inst->x_t[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	ASSERT(x0_existe);
	//
	if (x0_existe) {
		/*d_kerd_imaxmin<<<dim3(C0, KERD(GRAND_T,1)), dim3(1024,1)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,   inst->L,
			inst->y__d, inst->l__d,
			inst->dy__d,
			//
			mega_t,
			//
			C0
		);*/
		d_kerd_imaxmin<<<dim3(KERD(inst->x_Y[0]*C0,16), KERD(GRAND_T,16)), dim3(16,16)>>>(	// faire Min ET Max !!!
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,//  inst->L,
			inst->y__d,// inst->l__d,
			inst->dy__d,
			//
			mega_t,
			//
			C0
		);
	} else {
		// rien
	}
};