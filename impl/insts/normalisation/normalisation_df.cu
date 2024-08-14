#include "activation.cuh"

__global__
static void d_kerd__normalisation(
	uint x0_t, uint X0, float * x0, float * dx0,
	uint x1_t, uint X1, float * x1, float * dx1,
	//
	uint    Y,
	float * y,
	float * dy,
	//
	uint mega_t,
	//
	uint C0)
{
	uint _x = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_x < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint tx1 = t_MODE(_t, mega_t-x1_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		uint _c0 = (_x - (_x % (X0/C0)))/(X0/C0);
		//
		float _x0 = x0[tx0*X0 + _x];
		float _max = x1[tx1*2*C0 + _c0*2 + 0];
		float _min = x1[tx1*2*C0 + _c0*2 + 1];
		//
		//y[ty*Y + _x] = (_x0-_min)/(_max-_min);
		//l[ty*L + _x] = 1/(_max-_min);
		//
		float _dy = dy[ty*Y + _x];
		//
		float _dx0 = _dy / (_max - _min);
		float _dmin = _dy * (-1/(_max-_min) + (-1)*(-1)*(_x0-_min)/powf(_max-_min,2) );
		float _dmax = _dy * (  -(_x0-_min)/powf(_max-_min,2));
		//
		atomicAdd(&dx0[tx0*X0 + _x], _dx0);
		atomicAdd(&dx1[tx1*2*C0 + _c0*2 + 0], _dmax);
		atomicAdd(&dx1[tx1*2*C0 + _c0*2 + 1], _dmin);
	};
};

void normalisation__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint x0_t = inst->x_t[0];
	uint Y  = inst->Y;
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		d_kerd__normalisation<<<dim3(KERD(Y,16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			inst->x_t[1], inst->x_Y[1], x__d[1], dx__d[1],
			//
			inst->Y,
			inst->y__d,
			inst->dy__d,
			//
			mega_t,
			//
			inst->params[0]
		);
	} else {
		// rien
	}
};