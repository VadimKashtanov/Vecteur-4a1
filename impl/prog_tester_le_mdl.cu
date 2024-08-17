#include "main.cuh"

#include "../impl_template/tmpl_etc.cu"

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

static float ** toutes_les_predictions(Mdl_t * mdl, BTCUSDT_t * btcusdt) {
	//
	uint I = btcusdt->I;
	uint T = btcusdt->T;
	uint L = btcusdt->L;
	uint N = btcusdt->N;
	//
	float ancien_u = 100.0;
	float u = 100.0;
	//
	ASSERT(btcusdt->T % MEGA_T == 0);
	//
	uint _T     = (btcusdt->T - (btcusdt->T % MEGA_T))/MEGA_T;
	uint PREDS = _T * MEGA_T;
	//
	float * les_predictions = alloc<float>(PREDS);
	float * les_deltas      = alloc<float>(PREDS);
	float * les_prixs       = alloc<float>(PREDS);
	
	//
	uint lp = 0;
	//
	printf("[t=0] u = %f $\n", u);
	FOR(0, _t_, _T) {
		//
		uint ts[GRAND_T];
		FOR(0, t, GRAND_T) ts[t] = _t_*MEGA_T + 0;
		//
		uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);
		
		//
		mdl_f(mdl, btcusdt, ts__d);
		//
		uint Y    = mdl->inst[mdl->sortie]->Y;
		float * y = gpu_vers_cpu<float>(mdl->inst[mdl->sortie]->y__d, GRAND_T*MEGA_T*Y);
		//
		FOR(0, mega_t, MEGA_T) {
			uint ty = t_MODE(0, mega_t);
			//
			uint pos = _t_*MEGA_T + mega_t;
			float p0 = lire(btcusdt->prixs__d, _t_  );
			float p1 = (_t_ == _T-1 ? p0 : lire(btcusdt->prixs__d, _t_+1));
			//
			les_predictions[pos] = y[ty*Y + 0];
			les_deltas     [pos] = p1/p0 - 1.0;
			les_prixs      [pos] = p0;
			//
			u += u * 10 * les_predictions[pos] * les_deltas[pos];
			if (u < 0) u = 0;
		}

		//
		cudafree<uint>(ts__d);
		free(y);
		printf("[t=%i] u = %f $ ", 1+_t_, u);
		if      (ancien_u > u) printf("\033[91m-%.2g$\033[0m", abs(ancien_u-u));
		else if (ancien_u < u) printf("\033[92m+%.2g$\033[0m", abs(ancien_u-u));
		else                   printf("\033[2m  ?\033[0m");
		printf("\n");
		ancien_u = u;
	};
	//
	float ** ret = alloc<float*>(3);
	ret[0] = les_predictions;
	ret[1] = les_deltas     ;
	ret[2] = les_prixs      ;
	return ret;
};

int main() {
	srand(0);
	init_listes_instructions();
	ecrire_structure_generale("structure_generale.bin");
	verif_insts();

	//	=========================================================
	//	=========================================================
	//	=========================================================
	BTCUSDT_t * btcusdt = cree_btcusdt("prixs/tester_model_donnee.bin");

	//	=========================================================
	//	=========================================================
	//	=========================================================

	//	--- Mdl_t ---
	Mdl_t * mdl = ouvrire_mdl("mdl.bin");

	float ** __lp = toutes_les_predictions(mdl, btcusdt);
	float * preds  = __lp[0];
	float * deltas = __lp[1];
	float * prixs  = __lp[2];

	FILE * fp = FOPEN("les_predictions.bin", "wb");
	//
	uint T     = (btcusdt->T - (btcusdt->T % MEGA_T))/MEGA_T;
	uint PREDS = T * MEGA_T;
	//
	FWRITE(preds, sizeof(float), PREDS, fp);	//les prédictions
	free(preds);
	//
	FWRITE(deltas, sizeof(float), PREDS, fp);	//les déltas
	free(deltas);
	//
	FWRITE(prixs, sizeof(float), PREDS, fp);	//les déltas
	free(prixs);
	//
	fclose(fp);

	//	=========================================================
	//	=========================================================
	//	=========================================================
	//
	//plumer_le_score(mdl, btcusdt);

	//
	liberer_mdl    (mdl    );
	liberer_btcusdt(btcusdt);
};