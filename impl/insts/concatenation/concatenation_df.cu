#include "concatenation.cuh"

__global__
static void d_kerd__concatenation(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint mega_t,
	//
	uint Ax, uint Ay, uint Ay_c0, uint C0)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_y < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		uint __x = tx0*X0 + _y;
		//float _x = x0[__x];
		//
		uint c0 = (_y - (_y%(Ax*Ay*Ay_c0)))/(Ax*Ay*Ay_c0);
		_y -= c0*Ax*Ay*Ay_c0;
		uint c0_Ay_c0 = (_y - (_y%(Ax*Ay)))/(Ax*Ay);
		_y -= c0_Ay_c0*Ax*Ay;
		uint y_Ay = (_y - (_y%(Ax)))/(Ax);
		_y -= y_Ay*Ax;
		uint x_Ax = (_y - (_y%1))/1;
		_y -= x_Ax;
		//
		assert(_y == 0);
		//
		float _dy = dy[ty*Y + c0*Ax*Ay*Ay_c0 + y_Ay*(Ay_c0*Ax) + c0_Ay_c0*Ax + x_Ax];
		atomicAdd(&dx0[__x], _dy * 1);
	};
};

void concatenation__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint \
		Ax    = inst->params[0],	\
		Ay    = inst->params[1],	\
		Ay_c0 = inst->params[2],	\
		C0    = inst->params[3];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	if (x0_existe) {
		d_kerd__concatenation<<<dim3(KERD(X0,16), KERD(GRAND_T,8)), dim3(16,8)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			mega_t,
			//
			Ax, Ay, Ay_c0, C0
		);
	} else {
		//	rien
	}
};