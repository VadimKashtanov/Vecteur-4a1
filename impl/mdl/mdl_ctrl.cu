#include "mdl.cuh"

#include "../impl_template/tmpl_etc.cu"

void mdl_dy_zero(Mdl_t * mdl) {
	FOR(0, i, mdl->insts) {
		uint I = mdl->inst[i]->Y*GRAND_T*MEGA_T;
		kerd_liste_inis<float><<<dim3(KERD(I, 64)), dim3(64)>>>(
			mdl->inst[i]->dy__d, 0.0, I
		);
		if (mdl->inst[i]->P != 0) {
			uint P = mdl->inst[i]->P;
			kerd_liste_inis<float><<<dim3(KERD(P, 64)), dim3(64)>>>(
				mdl->inst[i]->dp__d, 0.0, P
			);
		}
	};
	ATTENDRE_CUDA();
};

static
__global__ void k64(float * l, float * l2, uint I) {
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	//
	uint t = threadIdx.x;
	//
	float __shared__ s[64];
	//64
	float a = (thx*2<I ? l[thx*2] : 0.0);
	float b = (thx*2+1<I ? l[thx*2+1] : 0.0);
	s[t] = MAX2(fabs(a), fabs(b));
	__syncthreads();
	//32
	if (t % 2 == 0) {
		s[t] = MAX2(s[t], s[t+1]);
	}
	__syncthreads();
	//16
	if (t % 4 == 0) {
		s[t] = MAX2(s[t], s[t+1*2]);
	}
	__syncthreads();
	//8
	if (t % 8 == 0) {
		s[t] = MAX2(s[t], s[t+1*2*2]);
	}
	__syncthreads();
	//4
	if (t % 16 == 0) {
		s[t] = MAX2(s[t], s[t+1*2*2*2]);
	}
	__syncthreads();
	//2
	if (t % 32 == 0) {
		s[t] = MAX2(s[t], s[t+1*2*2*2*2]);
	}
	__syncthreads();
	//1
	if (t % 64 == 0) {
		s[t] = MAX2(s[t], s[t+1*2*2*2*2*2]);
	}
	__syncthreads();
	//
	if (t % 64 == 0) {
		l2[thx/64] = s[t];
		//printf(">>%f\n", s[t]);
	}
};

/*static float* div_64(float * l, uint I) {
	float * l2 = cudalloc<float>(I);
	k64<<<dim3(KERD(DIV(I,2),64)), dim3(64)>>>(l, l2, I);
	ATTENDRE_CUDA();
	return l2;
};

static float max_abs_grad_inst(float * l, uint I) {
	if (I != 0) {
		float * l1 = l;
		float nb = logf(I) / logf(64.0);
		FOR(0, i, nb) {
			float * l2 = div_64(l1, I);
			I = KERD(I,64)*64 / 64;
			if (i == 0) {
				l1 = l2;
			} else {
				cudafree<float>(l1);
				l1 = l2;
			}
			if (I == 1) {
				float * _l2 = gpu_vers_cpu<float>(l2, I);
				cudafree<float>(l2);
				float ret = _l2[0];
				free(_l2);
				return ret;
			}
		};
	} else {
		return 0.0;
	}
};*/

float mdl_max_abs_grad(Mdl_t * mdl) {
	float _MAX[mdl->insts];
	uint     I[mdl->insts];
	float   nb[mdl->insts];
	FOR(0, i, mdl->insts) {
		_MAX[i] = 0.0;
		I[i]    = mdl->inst[i]->P;
		nb[i]   = logf(I[i]) / logf(64.0);
	}
	//
	float nb_max = nb[0];
	FOR(0, i, mdl->insts) if (nb_max < nb[i]) nb_max = nb[i];
	//
	float * l1[mdl->insts];
	float * l2[mdl->insts];
	FOR(0, i, mdl->insts) {
		l1[i] = mdl->inst[i]->dp__d;
	}
	//
	FOR(0, _nb, nb_max) {
		FOR(0, i, mdl->insts) {
			if (I[i] > 1) {
				//float * l2 = div_64(l1, I);
				//
				l2[i] = cudalloc<float>(I[i]);
				k64<<<dim3(KERD(DIV(I[i],2), 64)), dim3(64)>>>(l1[i], l2[i], I[i]);
				I[i] = KERD(I[i],64)*64 / 64;
			}
		}
		ATTENDRE_CUDA();
		//
		FOR(0, i, mdl->insts) {
			if (I[i] >= 1) {
				//
				if (_nb == 0) {
					l1[i] = l2[i];
				} else {
					cudafree<float>(l1[i]);
					l1[i] = l2[i];
				}
				if (I[i] == 1) {
					float * _l1 = gpu_vers_cpu<float>(l1[i], I[i]);
					cudafree<float>(l1[i]);
					_MAX[i] = _l1[0];
					//printf("---%f\n", _l1[0]);
					free(_l1);
					I[i] = 0;
				}
			}
		};
	}
	//
	float _max = _MAX[0];
	//
	FOR(1, i, mdl->insts) {
		if (_max < _MAX[i]) _max = _MAX[i];
	}
	//printf(">>>> %f\n", _max);
	//
	return _max;
};