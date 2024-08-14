#include "transpose2d.cuh"

static __global__ void kerd__transpose2d__simple(
	uint x0_t, uint X0, float * x0,
	//
	uint    Y,
	float * y,
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

	//	A faire :
	//	1) Regler la Transpose2d
	//	2) Ajouter des Scaled ( 1/sqrt(Dk) ) pour reajuster le attention mechanisme
	//	3) Ajouter des Masques au Attention (ou pas)
	//	4) Eventuellement jouer avec le softmax du attention, ou le normer autrement.
	//	5) Pas besoin de Norme car tous mes vecteurs seront toujours dans [-1;+1] peut importe où


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
		//printf("%i %i %i %i\n", _ax, _ay, _c0, __t);
		//
		y[At] = x0[A];
	}
};

//	---------------------------------------------------------------------------------

void transpose2d__f(Inst_t * inst, float ** x__d, uint * ts__d, uint mega_t) {
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
		kerd__transpose2d__simple<<<dim3(KERD((Ax*Ay),16), KERD((C0*GRAND_T),16)), dim3(16,16)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0],
			//
			inst->Y,
			inst->y__d,
			//
			ts__d, mega_t,
			//
			Ax, Ay, C0
		);
	} else {
		inst_zero_mega_t(inst, mega_t);
	}
};