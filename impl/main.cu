#include "main.cuh"

#include "../impl_template/tmpl_etc.cu"

static void cree_mdl_depuis_pre_mdl(BTCUSDT_t * btcusdt) {
	Mdl_t * mdl = cree_mdl_depuis_st_bin("mdl.st.bin");
	mdl_verif(mdl, btcusdt);
	ecrire_mdl("mdl.bin", mdl);
	liberer_mdl(mdl);
};

void montrer_Y_du_model_temporel(Mdl_t * mdl, BTCUSDT_t * btcusdt) {
	uint ts[GRAND_T];
	FOR(0, t, GRAND_T)
		ts[t] = rand() % (btcusdt->T - MEGA_T - 1);
	uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);
	//
	//mdl_allez_retour(mdl, btcusdt, ts__d);
	mdl_f(mdl, btcusdt, ts__d);
	//
	printf(" ======= Plumer Y ======\n");
	printf("mega_t = | ");
	FOR(0, i, MIN2(MEGA_T, 19)) printf("    %i   |", i);
	printf("\n");
	FOR(0, i, mdl->insts)
	{
		Inst_t * inst = mdl->inst[i];
		printf("#%i -- ID=%i %s Y=%i --\n", i, inst->ID, inst_Nom[inst->ID], inst->Y);
		//
		float * y = gpu_vers_cpu<float>(inst->y__d, inst->Y * GRAND_T * MEGA_T);
		//
		FOR(0, j, inst->Y) {
			printf("%i| ", j);
			FOR(0, mega_t, MIN2(MEGA_T, 19)) {
				uint ty = t_MODE(0, mega_t);
				printf("%+f ", y[ty*inst->Y + j]);
				if (y[ty*inst->Y + j] != y[ty*inst->Y + j]) ERR("Nan");
			}
			printf("\n");
		}
		//
		free(y);
	};
	//
	cudafree<uint>(ts__d);
};

void montrer_Y_du_model_sans_temps(Mdl_t * mdl, BTCUSDT_t * btcusdt) {
	uint ts[GRAND_T];
	FOR(0, t, GRAND_T)
		ts[t] = rand() % (btcusdt->T - MEGA_T - 1);
	uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);
	//
	//mdl_allez_retour(mdl, btcusdt, ts__d);
	mdl_f(mdl, btcusdt, ts__d);
	//
	printf(" ======= Plumer Y ======\n");
	FOR(0, i, mdl->insts)
	{
		Inst_t * inst = mdl->inst[i];
		printf("#%i -- ID=%i %s Y=%i --\n", i, inst->ID, inst_Nom[inst->ID], inst->Y);
		//
		float * y = gpu_vers_cpu<float>(inst->y__d, inst->Y * GRAND_T * MEGA_T);
		//
		FOR(0, j, inst->Y) {
			uint ty = t_MODE(0, 0);
			printf("%+f,", y[ty*inst->Y + j]);
			if (y[ty*inst->Y + j] != y[ty*inst->Y + j]) ERR("Nan");
		}
		printf("\n");
		//
		free(y);
	};
	//
	cudafree<uint>(ts__d);
};

int main() {
	srand(time(NULL));
	init_listes_instructions();
	ecrire_structure_generale("structure_generale.bin");
	verif_insts();

	//	=========================================================
	//	=========================================================
	//	=========================================================
	//verif_mdl_1e5();

	//exit(0);

	//	=========================================================
	//	=========================================================
	//	=========================================================
	BTCUSDT_t * btcusdt = cree_btcusdt("prixs/dar.bin");

	//	=========================================================
	//	=========================================================
	//	=========================================================

	//visualiser_vitesses("mdl.bin", btcusdt);

	//	=========================================================
	//	=========================================================
	//	=========================================================

	//	--- Re-cree le Model ---
	cree_mdl_depuis_pre_mdl(btcusdt);

	//	--- Mdl_t ---
	Mdl_t * mdl = ouvrire_mdl("mdl.bin");
	plumer_model(mdl);
	//montrer_Y_du_model_temporel(mdl, btcusdt);
	//montrer_Y_du_model_sans_temps(mdl, btcusdt);
	//tester_le_model(mdl, btcusdt);

	//	=========================================================
	//	=========================================================
	//	=========================================================
	uint un_mois = ((24*30 - (24*30 % MEGA_T)) / MEGA_T) * MEGA_T;
	//
//plumer_le_score(mdl, btcusdt);
	// 
	uint e = 0;       // Atention Mechanisme, alternative Dot1d AB, ...
	while (true) {
		printf(" === Echope %i ===\n", e);
		
		//
		uint I        = 50;
		uint tous_les = 10;

		//
		srand(time(NULL));
		uint ts[GRAND_T];
		FOR(0, t, GRAND_T)
			ts[t] = rand() % (btcusdt->T - MEGA_T - 1 - un_mois);
		uint * ts__d = cpu_vers_gpu<uint>(ts, GRAND_T);

		//
		opti(
			mdl, btcusdt,
			ts__d,
			I,
			tous_les,
			ADAM, 5e-4
		);
		ecrire_mdl("mdl.bin", mdl);

		if (e % 10 == 0) {
			printf("pause ...\n");
			sleep(2);
		}
		
		//
		e++;

		//
		cudafree<uint>(ts__d);
	}

	//
	//liberer_mdl    (mdl    );
	//liberer_btcusdt(btcusdt);
};