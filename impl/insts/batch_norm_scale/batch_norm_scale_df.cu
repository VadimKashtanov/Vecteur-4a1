#include "batch_norm_scale.cuh"

static __global__ void d_kerd_batch_norm_scale(
	uint x0_t, uint X0, float * x0, float * dx0,
	uint x1_t, uint X1, float * x1, float * dx1,
	uint x2_t, uint X2, float * x2, float * dx2,
	//
	float * p, float * dp,
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
	if (_x < X0 && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		//uint tx1 = t_MODE(_t, mega_t-x1_t);
		//uint tx2 = t_MODE(_t, mega_t-x2_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		float s = x0[tx0*X0 + _x];
		//
		uint c0 = (_x - (_x%(X0/C0))  )/(X0/C0);
		//
		float miu = x1[0*X1 + c0];
		float var = x2[0*X2 + c0];
		//
		float alpha = p[c0*2 + 0];
		//float beta  = p[c0*2 + 1];
		//
		float _dx = alpha * 1 / sqrtf(var + 1e-8);
		float _dm = -_dx;
		float _dv = -alpha * (s - miu) / (2*powf(var + 1e-8, 1.5));
		//
		float dalpha = (s - miu) / sqrtf(var + 1e-8);
		float dbeta  = 1;
		//
		float _dy = dy[ty*Y + _x];
		//
		atomicAdd(&dx0[tx0*X0 + _x], _dy * _dx);
		atomicAdd(&dx1[0  *X1 + c0], _dy * _dm);
		atomicAdd(&dx2[0  *X2 + c0], _dy * _dv);
		//
		atomicAdd(&dp [c0*2 + 0], _dy * dalpha);
		atomicAdd(&dp [c0*2 + 1], _dy * dbeta );
	};
}

void batch_norm_scale__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint \
		C0 = inst->params[0];
	//
	uint x0_t = inst->x_t[0];
	uint x1_t = inst->x_t[1];
	uint x2_t = inst->x_t[2];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	bool x1_existe = (mega_t != 0 ? true : (x1_t != 1));
	bool x2_existe = (mega_t != 0 ? true : (x2_t != 1));
	//
	ASSERT(x0_existe && x1_existe && x2_existe);
	//
	if (x0_existe && x1_existe && x2_existe) {
		d_kerd_batch_norm_scale<<<dim3(KERD(inst->x_Y[0],16), KERD(GRAND_T,16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			inst->x_t[1], inst->x_Y[1], x__d[1], dx__d[1],
			inst->x_t[2], inst->x_Y[2], x__d[2], dx__d[2],
			//
			inst->p__d, inst->dp__d,
			//
			inst->Y,
			inst->y__d,
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