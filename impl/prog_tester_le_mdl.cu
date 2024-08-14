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
	float u = 100.0;
	//
	ASSERT(btcusdt->T % MEGA_T == 0);
	//
	uint T     = (btcusdt->T - (btcusdt->T % MEGA_T))/MEGA_T;
	uint PREDS = T * MEGA_T;
	//
	float * les_predictions = alloc<float>(PREDS);
	float * les_deltas      = alloc<float>(PREDS);
	
	//
	uint lp = 0;
	//
	FOR(0, _t_, T) {
		//
		uint ts[GRAND_T];
		FOR(0, t, GRAND_T) ts[t] = _t_*MEGA_T + 0;
		//
		uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);
		
		//
		mdl_f(mdl, btcusdt, ts__d);
		//
		uint Y = mdl->inst[mdl->inst_sortie]->Y;
		float * y = gpu_vers_cpu<float>(mdl->inst[mdl->sortie]->y__d, GRAND_T*MEGA_T*Y);
		FOR(0, mega_t, MEGA_T) {
			uint ty = t_MODE(0, mega_t);
			les_predictions[lp] = y[ty*Y + 0];
			les_deltas     [lp] = 0;//lire(btcusdt->sorties__d, (ts[0] + mega_t)*btcusdt->Y+0);
			lp++;

			u += u*1*les_predictions[lp-1]*les_deltas[lp-1];
		}

		//
		cudafree<uint>(ts__d);
		free(y);
		printf("u = %f $\n", u);
	};
	//
	float ** ret = alloc<float*>(2);
	ret[0] = les_predictions;
	ret[1] = les_deltas     ;
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
	float * lp = __lp[0];
	float * dl = __lp[1];

	FILE * fp = FOPEN("les_predictions.bin", "wb");
	//
	uint T     = (btcusdt->T - (btcusdt->T % MEGA_T))/MEGA_T;
	uint PREDS = T * MEGA_T;
	//
	FWRITE(lp, sizeof(float), PREDS, fp);	//les prédictions
	free(lp);
	//
	FWRITE(dl, sizeof(float), PREDS, fp);	//les déltas
	free(dl);
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