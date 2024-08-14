#include "mdl.cuh"

#include "../impl_template/tmpl_etc.cu"

__global__
static void kerd_p1e5(float * p__d, uint p, float _1E5) {
	p__d[p] += _1E5;
};

static void plus_1e5(float * p__d, uint p, float _1E5) {
	kerd_p1e5<<<1,1>>>(p__d, p, _1E5);
	ATTENDRE_CUDA();
};

//	---------------------------------------------------

__global__
static void kerd_lire(float * p__d, uint p, float * val) {
	val[0] = p__d[p];
};

static float lire(float * p__d, uint p) {
	float * val = cudalloc<float>(1);
	kerd_lire<<<1,1>>>(p__d, p, val);
	ATTENDRE_CUDA();
	//
	float * _ret = gpu_vers_cpu<float>(val, 1);
	float ret = _ret[0];
	free(_ret);cudafree<float>(val);
	//
	return ret;
};

void tester_le_model(Mdl_t * mdl, BTCUSDT_t * btcusdt) {
	uint ts[GRAND_T];
	FOR(0, t, GRAND_T) ts[t] = rand() % (btcusdt->T - MEGA_T);
	uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);
	//
	mdl_verif(mdl, btcusdt);
	//
	//
	mdl_allez_retour(mdl, btcusdt, ts__d);
	float * grad_cuda[mdl->insts];
	FOR(0, i, mdl->insts) {
		if (mdl->inst[i]->P > 0)
			grad_cuda[i] = gpu_vers_cpu<float>(mdl->inst[i]->dp__d, mdl->inst[i]->P);
	}
	//
	//
	INIT_CHRONO(s)
	DEPART_CHRONO(s)
	//
	float S = mdl_S(mdl, btcusdt, ts__d);
	//
	float _1E5 = 3e-2;//5e-3;
	uint lp = 0;
	FOR(0, i, mdl->insts) {
		printf("#### INSTRUCTION %i (%s Y=%i) ####\n",
			i, 
			inst_Nom[mdl->inst[i]->ID], mdl->inst[i]->Y
		);
		//
		FOR(0, p, mdl->inst[i]->P) {

			//	f(x + 1e-5)
			plus_1e5(mdl->inst[i]->p__d, p, +_1E5);
			float S1e5 = mdl_S(mdl, btcusdt, ts__d);
			plus_1e5(mdl->inst[i]->p__d, p, -_1E5);

			//	df
			float a = (S1e5 - S)/_1E5;

			//	f'
			float b = grad_cuda[i][p];

			//	vitesse
			float vitesse = (float)(++lp) / VALEUR_CHRONO(s);

			//
			printf("%i| ", p);
			PLUME_CMP(a, b);
			if (b != 0) printf(" (x%+f) ", a/b);
			printf(" (%+f m/s)   inst=%i\n", vitesse, i);
		};
	};
	printf("1E5  === dp\n");
	//
	FOR(0, i, mdl->insts) if (mdl->inst[i]->P > 0) free(grad_cuda[i]);
	//
	cudafree<uint>(ts__d);
};