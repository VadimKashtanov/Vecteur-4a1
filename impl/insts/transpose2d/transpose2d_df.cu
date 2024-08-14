#include "transpose2d.cuh"

static __global__ void d_kerd__transpose2d__simple(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint * ts__d, uint mega_t,
	//
	uint Ax, uint Ay, uint C0)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	uint thy = threadIdx.y + blockIdx.y * blockDim.y;

	//	thx = Ax*Ay
	uint _ax = thx % Ax;
	uint _ay = (thx-_ax)/Ax;

	//	thy = C0*GRAND_T
	uint _c0 = thy % C0;
	uint __t = (thy-_c0)/C0;

	if (_ay < Ay && _ax < Ax && _c0 < C0 && __t < GRAND_T) {
		uint tx0 = t_MODE(__t, mega_t-x0_t);
		uint ty  = t_MODE(__t, mega_t     );
		//
		uint A  = tx0*X0 + _c0*(Ax*Ay) + _ay*Ax + _ax;
		uint At = ty*Y   + _c0*(Ax*Ay) + _ax*Ay + _ay;
		//
		//y[At] = x0[A];
		atomicAdd(&x0[A], y[At]);
	}
};

//	---------------------------------------------------------------------------------

void transpose2d__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint * params = inst->params;
	uint \
		Ax =params[0],	\
		Ay =params[1],	\
		C0 =params[2];
	//
	uint x0_t = inst->x_t[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	ASSERT(x0_existe);
	//
	if (x0_existe) {
		d_kerd__transpose2d__simple<<<dim3(KERD((Ax*Ay),16), KERD((C0*GRAND_T),16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			ts__d, mega_t,
			//
			Ax, Ay, C0
		);
	} else {
		//inst_zero_mega_t(inst, mega_t);
	}
};