#include "matmul_poid_PA.cuh"

static __global__ void d_kerd__matmul_poid_PA__simple(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	float * p, float * dp,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint * ts__d, uint mega_t,
	//
	uint Ax, uint Ay, uint Bx, uint C0)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	uint thy = threadIdx.y + blockIdx.y * blockDim.y;

	//	thx = Ay*C0
	uint _ay = thx % Ay;
	uint _c0 = (thx-_ay)/Ay;

	//	thy = Bx*GRAND_T
	uint _bx = thy % Bx;
	uint  _t = (thy-_bx)/Bx;

	if (_ay < Ay && _c0 < C0 && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		uint pos_y = ty*Y + _c0*(Bx*Ay) + _ay*(Bx) + _bx;
		float _dy = dy[pos_y];
		//
		FOR(0, k, Ax) {
			uint pos_p  =                _c0*(Ax*Ay) + _ay*Ax + k;
			uint pos_x0 = tx0*C0*Bx*Ax + _c0*(Bx*Ax) + k*Bx + _bx;
			//
			//s += x0[pos_x0] * x1[pos_x0];
			atomicAdd(&dx0[pos_x0], p [pos_p ] * _dy);
			atomicAdd(&dp [pos_p ], x0[pos_x0] * _dy);
		}
	}
};

//	---------------------------------------------------------------------------------

void matmul_poid_PA__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint * params = inst->params;
	uint \
		Ax =params[0],	\
		Ay =params[1],	\
		Bx =params[2],	\
		C0 =params[3];
	//
	uint x0_t = inst->x_t[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	//ASSERT(x0_existe);
	//
	if (x0_existe) {
		d_kerd__matmul_poid_PA__simple<<<dim3(KERD((Ay*C0),16), KERD((Bx*GRAND_T),16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->p__d, inst->dp__d,
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			ts__d, mega_t,
			//
			Ax, Ay, Bx, C0
		);
	} else {
		//inst_zero_mega_t(inst, mega_t);
	}
};