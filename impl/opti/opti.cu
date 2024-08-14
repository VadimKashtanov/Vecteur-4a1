#include "opti.cuh"

#include "../impl_template/tmpl_etc.cu"

uint hists[] = {
	SGD_____HISTOIRE,
	MOMENT__HISTOIRE,
	RMSPROP_HISTOIRE,
	ADAM____HISTOIRE
};

void opti(
	Mdl_t     *     mdl,
	BTCUSDT_t * btcusdt,
	uint      *   ts__d,
	uint              I,
	uint       tous_les,
	uint        methode,
	float         alpha)
{
	//	--- Hist ---
	float *** hist = alloc<float**>(hists[methode]);
	FOR(0, h, hists[methode]) {
		hist[h] = alloc<float*>(mdl->insts);
		FOR(0, i, mdl->insts) {
			hist[h][i] = cudalloc<float>(mdl->inst[i]->P);
			// = 0
		}
	}

	//	--- Plume ---
	mdl_plume_grad(mdl, btcusdt, ts__d);
	//
	float _max_abs_grad = 1;//mdl_max_abs_grad(mdl);
	if (_max_abs_grad == 0) ERR("Le grad max est = 0");
	//
	alpha /= _max_abs_grad;
	//
	printf("alpha=%f, max_abs_grad=%f => nouveau alpha=%f\n", alpha, _max_abs_grad, alpha / _max_abs_grad);
	//
	//	--- Opti  ---
	FOR(0, i, I) {
		if (i != 0) {
			//	dF(x)
			mdl_allez_retour(mdl, btcusdt, ts__d);

			//	x = x - dx
			if (methode == SGD    ) sgd    (mdl, hist, i, alpha, i);
			if (methode == MOMENT ) moment (mdl, hist, i, alpha, i);
			if (methode == RMSPROP) rmsprop(mdl, hist, i, alpha, i);
			if (methode == ADAM   ) adam   (mdl, hist, i, alpha, i);
		}
		//
		if (i % tous_les == 0) {
			float s = mdl_S(mdl, btcusdt, ts__d);
			//
			float* p0 = pourcent_btcusdt(btcusdt, mdl->inst[mdl->sortie]->y__d, ts__d);
			//
			printf("%3.i/%3.i score = %f inconnue=(", i,I,s);
			FOR(0, j, INTERVS) printf("%f%%,", roundf(p0[j]*100*1000)/1000);
			printf(") connue=(");
			FOR(0, j, INTERVS) printf("%f%%,", roundf(p0[INTERVS+j]*100*1000)/1000);
			printf(")\n");
		};
	};
	//
	//
	FOR(0, h, hists[methode]) {
		FOR(0, i, mdl->insts) {
			cudafree<float>(hist[h][i]);
		}
		free(hist[h]);
	}
	free(hist);
}