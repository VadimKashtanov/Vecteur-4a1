#include "matmul_poid_AP.cuh"

static __global__ void kerd__matmul_poid_AP__simple(
	uint x0_t, uint X0, float * x0,
	//
	float * p,
	//
	uint    Y,
	float * y,
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
		float s = 0;
		uint pos_y = ty*Y + _c0*(Bx*Ay) + _ay*(Bx) + _bx;
		FOR(0, k, Ax) {
			uint pos_x0 = tx0*C0*Ax*Ay + _c0*(Ax*Ay) + _ay*Ax + k;
			uint pos_p  =                _c0*(Bx*Ax) + k*Bx + _bx;
			//
			s += x0[pos_x0] * p[pos_p];
		}
		y[pos_y] = s;
	}
};

//	---------------------------------------------------------------------------------

void matmul_poid_AP__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
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
		kerd__matmul_poid_AP__simple<<<dim3(KERD((Ay*C0),16), KERD((Bx*GRAND_T),16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->p__d,
			//
			inst->Y,
			inst->y__d,
			//
			ts__d, mega_t,
			//
			Ax, Ay, Bx, C0
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};