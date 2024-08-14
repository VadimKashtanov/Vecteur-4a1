#include "mdl.cuh"

float mdl_S(Mdl_t * mdl, BTCUSDT_t * btcusdt, uint * ts__d) {
	mdl_f(mdl, btcusdt, ts__d);
	//
	mdl_dy_zero(mdl);
	//
	return f_df_btcusdt(
		btcusdt,
		mdl->inst[mdl->sortie]->y__d,
		mdl->inst[mdl->sortie]->dy__d,
		ts__d);
};