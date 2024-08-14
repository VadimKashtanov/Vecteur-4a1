#include "softmax.cuh"

__global__
static void d_kerd__softmax(
	uint x0_t, uint X0, float * x0, float * dx0,
	//
	uint    Y,
	float * y, float * dy,
	//
	uint mega_t,
	//
	uint Vect)
{
	uint _v = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (_v < Y && _t < GRAND_T) {
		uint tx0 = t_MODE(_t, mega_t-x0_t);
		uint ty  = t_MODE(_t, mega_t     );
		//
		/*FOR(0, i, Vect) {
			float _dx0 = 0;
			FOR(0, j, Vect) {
				//float val = expf(x0[tx0*X0 + _v*Vect + i]) / somme;
				//y[tx0*X0 + _v*Vect + i] = expf(x0[tx0*X0 + _v*Vect + i]) / somme;
				//if (i == j) atomicAdd(&dx0[tx0*X0 + _v*Vect + i], y[tx0*X0 + _v*Vect + i]);
				//else        atomicAdd(&dx0[tx0*X0 + _v*Vect + i], y[tx0*X0 + _v*Vect + i]);
				float delta = (float)(i == j);
				_dx0 += y[ty*X0 + _v*Vect + j]*(delta - y[ty*X0 + _v*Vect + i]) * dy[ty*X0 + _v*Vect + j];
			}
			atomicAdd(&dx0[tx0*X0 + _v*Vect + i], _dx0);
		}*/
		float somme = 0;
		FOR(0, i, Vect) somme += expf(x0[tx0*X0 + _v*Vect + i]);
		//
		float d_somme = 0;
		FOR(0, i, Vect) {
			//y[ty*X0 + _v*Vect + i] = expf(x0[tx0*X0 + _v*Vect + i]) / somme;
			d_somme += dy[ty*X0 + _v*Vect + i] * expf(x0[tx0*X0 + _v*Vect + i]) / (somme*somme) * (-1);
			atomicAdd(&dx0[tx0*X0 + _v*Vect + i], dy[ty*X0 + _v*Vect + i] * expf(x0[tx0*X0 + _v*Vect + i]) / somme);
		}
		FOR(0, i, Vect) {
			//somme += expf(x0[tx0*X0 + _v*Vect + i]);
			atomicAdd(&dx0[tx0*X0 + _v*Vect + i], d_somme * expf(x0[tx0*X0 + _v*Vect + i]));
		}
	};
};

void softmax__df(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t) {
	uint X0 = inst->x_Y[0]; uint x0_t = inst->x_t[0];
	//
	uint \
		Vect    = inst->params[0];
	//
	bool x0_existe = (mega_t != 0 ? true : (x0_t != 1));
	//
	uint Vects = X0/Vect;
	//
	if (x0_existe) {
		d_kerd__softmax<<<dim3(KERD(Vects,16), KERD(GRAND_T,8)), dim3(16,8)>>>(
			inst->x_t[0], inst->x_Y[0], x__d[0], dx__d[0],
			//
			inst->Y,
			inst->y__d, inst->dy__d,
			//
			mega_t,
			//
			Vect
		);
	} else {
		//	rien
	}
};