#include "mdl.cuh"

float* mdl_pourcent(Mdl_t * mdl, BTCUSDT_t * btcusdt, uint * ts__d) {
	mdl_f(mdl, btcusdt, ts__d);
	return pourcent_btcusdt(btcusdt, mdl->inst[mdl->sortie]->y__d, ts__d);
};