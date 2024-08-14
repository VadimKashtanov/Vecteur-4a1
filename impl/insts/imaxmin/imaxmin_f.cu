#include "imaxmin.cuh"

#include "../impl_template/tmpl_etc.cu"

#define MAX_float 123456789.87654321

/*static __global__ void kerd_imaxmin(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y, uint    L,
	float * y, float * l,
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
		//
		__shared__ float        max[1024];
		__shared__ float        min[1024];
		__shared__ uint origine_max[1024];
		__shared__ uint origine_min[1024];
		//
		uint _X = X0 / C0;
		//
		uint _origine = 0*1024 + thx;
		float _val = (_origine<_X ? x0[tx0*X0 + _c0*_X + _origine] : MAX_float);
		//
		float _max = _val;
		float _min = _val;
		uint _origine_max = _origine;
		uint _origine_min = _origine;
		//
		FOR(1, i, KERD(_X, 1024)) {
			uint p = i*1024+thx;
			if (p < _X) {
				_val = x0[tx0*X0 + _c0*_X + p];
				//
				if (_val > _max) {
					_max = _val;
					_origine_max = p;
				};
				if (_val < _min) {
					_min = _val;
					_origine_min = p;
				};
			};
		}
		max[thx] = _max;
		min[thx] = _min;
		origine_max[thx] = _origine_max;
		origine_min[thx] = _origine_min;
		//
		uint thx_a, thx_b;
		float max_a, max_b, min_a, min_b;
		uint omax_a, omax_b, omin_a, omin_b;
		__syncthreads();
		//
		FOR(0, i, 10) {
			uint modulo = pow(2, 1 + i);
			if (!(modulo >= 2 && modulo <= 1024)) {
				printf("modulo = %i\n", modulo);
				assert(0);
			}
			if (thx % modulo == 0) {
				//
				thx_a = thx +   0    ;
				thx_b = thx +modulo-1;
				//
				max_a  = max        [thx_a]; max_b  = max        [thx_b];
				omax_a = origine_max[thx_a]; omax_b = origine_max[thx_b];
				if (max_a == MAX_float) {
					max[thx] = max_b;
					origine_max[thx] = omax_b;
				} else if (max_b == MAX_float) {
					max[thx] = max_a;
					origine_max[thx] = omax_a;
				} else if (max_a > max_b) {
					max[thx] = max_b;
					origine_max[thx] = omax_b;
				} else {
					max[thx] = max_a;
					origine_max[thx] = omax_a;
				}
				//
				min_a  = min        [thx_a]; min_b  = min        [thx_b];
				omin_a = origine_min[thx_a]; omin_b = origine_min[thx_b];
				if (min_a == MAX_float) {
					min[thx] = min_b;
					origine_min[thx] = omin_b;
				} else if (min_b == MAX_float) {
					min[thx] = min_a;
					origine_min[thx] = omin_a;
				} else if (min_a > min_b) {
					min[thx] = min_b;
					origine_min[thx] = omin_b;
				} else {
					min[thx] = min_a;
					origine_min[thx] = omin_a;
				}
				//
			}
			__syncthreads();
		}
		//
		y[ty*2*C0 + 2*_c0 + 0] = max[0];
		y[ty*2*C0 + 2*_c0 + 1] = min[0];
		l[ty*2*C0 + 2*_c0 + 0] = (float)origine_max[0];
		l[ty*2*C0 + 2*_c0 + 1] = (float)origine_min[0];
		printf("_c0=%i %f %f  (%f %f %f %f)\n", _c0, max[0], min[0],   max[0],max[1],max[2],max[3]);
	};
}*/

static __global__ void kerd_imaxmin(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,// uint    L,
	float * y,// float * l,
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
		atomicMax(&y[ty*Y + _c0*2+0], __x);
		atomicMin(&y[ty*Y + _c0*2+1], __x);
	}
}

void imaxmin__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
	uint \
		C0     = inst->params[0];
	//
	uint x0_t = inst->x_t[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	ASSERT(x0_existe);
	//
	inst_zero_mega_t(inst, mega_t);
	//
	if (x0_existe) {
		/*kerd_imaxmin<<<dim3(C0, KERD(GRAND_T,1)), dim3(1024,1)>>>(	// faire Min ET Max !!!
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y,  inst->L,
			inst->y__d, inst->l__d,
			//
			mega_t,
			//
			C0
		);*/
		kerd_imaxmin<<<dim3(KERD(inst->x_Y[0]*C0,16), KERD(GRAND_T,16)), dim3(16,16)>>>(	// faire Min ET Max !!!
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y,//  inst->L,
			inst->y__d,// inst->l__d,
			//
			mega_t,
			//
			C0
		);
	} else {
		//inst_zero_mega_t(inst, mega_t);
	}
};