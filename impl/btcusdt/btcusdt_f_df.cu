#include "btcusdt.cuh"

#include "../../impl_template/tmpl_etc.cu"

static __global__ void k__f_df_btcusdt(
	float * S,
	//
	float * y, float * dy,
	float * w,
	//
	uint * ts__d,
	//
	uint I, uint T, uint L, uint N)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x; 
	//uint _t = threadIdx.y + blockIdx.y * blockDim.y;
	//uint  i = threadIdx.z + blockIdx.z * blockDim.z;
	//
	if (_y < (L*N)) {
		float s = 0;
		FOR(0, i, I) {
			FOR(0, _t, GRAND_T) {
				FOR(0, mega_t, MEGA_T) {
					uint ty        = t_MODE(_t, mega_t);
					uint t_btcusdt = ts__d[_t] + 1 + mega_t;
					assert(t_btcusdt < T);
					//
					float __y = y[ty*I*L*N + i*L*N + _y];
					float __w = w[ i*T*L*N + t_btcusdt*L*N + _y];
					//printf("i=%i _y=%i pos=%i %f %f\n", i, _y, i*L*N + _y, __y, __w);
					if (__y != __y) {
						printf("i=%i _y=%i pos=%i %f %f\n", i, _y, i*L*N + _y, __y, __w);
						assert(0);
					}
					assert(__y >= -100 && __y <= +100);
					//
					float coef = (float)(GRAND_T * MEGA_T * (I*L*N));
					s       += ( score_p2(__y, __w, 2)) / coef * (_y==0 ? 5:1);
					float ds = (dscore_p2(__y, __w, 2)) / coef * (_y==0 ? 5:1);
					//
					//atomicAdd(&dy[ty*I*N*L + i*L*N + _y], ds);
					dy[ty*I*N*L + i*L*N + _y] = ds;
				}
			}
		}
		//
		//atomicAdd(&S[0], s);
		S[_y] = s;
	}
};

float f_df_btcusdt(BTCUSDT_t * btcusdt, float * y__d, float * dy__d, uint * ts__d) {
	uint I=btcusdt->I;
	uint L=btcusdt->L;
	uint N=btcusdt->N;
	uint T=btcusdt->T;
	//
	float * S__d = cudalloc<float>(L*N);
	k__f_df_btcusdt<<<dim3(KERD((L*N), 32)/*,KERD(GRAND_T, 8), *//*KERD(I,4)*/), dim3(32/*,8,*//*4*/)>>>(
		S__d,
		y__d, dy__d,
		btcusdt->serie__d,
		ts__d,
		I, T, L, N
	);
	ATTENDRE_CUDA();
	//
	//
	float * S = gpu_vers_cpu<float>(S__d, L*N);
	float score = 0;
	FOR(0, i, L*N) score += S[i];
	//
	cudafree<float>(S__d);
	    free       (S   );
	//
	return score;
};