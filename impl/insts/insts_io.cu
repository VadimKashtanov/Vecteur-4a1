#include "insts.cuh"

#include "../../impl_template/tmpl_etc.cu"

static Inst_t * lire_tete_instruction(FILE * fp) {
	Inst_t * ret = alloc<Inst_t>(1);

	//
	FREAD(&ret->ID, sizeof(uint), 1, fp);
	if (!(ret->ID < INSTS)) ERR("ret->ID < INSTS  (ret->ID=%i INSTS=%i)", ret->ID, INSTS);
	
	//
	FOR(0, __x, inst_Xs[ret->ID]) {
		FREAD(&ret->x_Y  [__x], sizeof(uint), 1, fp);
		FREAD(&ret->x_pos[__x], sizeof(uint), 1, fp);
		FREAD(&ret->x_t  [__x], sizeof(uint), 1, fp);
	}
	
	//
	FREAD(&ret->Y, sizeof(uint), 1, fp);
	
	//
	FREAD(ret->params, sizeof(uint), inst_PARAMS[ret->ID], fp);
	
	//
	ret->P = calculer_P[ret->ID](ret->x_Y, ret->x_pos, ret->x_t, ret->Y, ret->params);
	ret->L = calculer_L[ret->ID](ret->x_Y, ret->x_pos, ret->x_t, ret->Y, ret->params);

	return ret;
};

static void ecrire_tete_instruction(FILE * fp, Inst_t * ret) {
	//
	FWRITE(&ret->ID, sizeof(uint), 1, fp);
	
	//
	FOR(0, __x, inst_Xs[ret->ID]) {
		FWRITE(&ret->x_Y  [__x], sizeof(uint), 1, fp);
		FWRITE(&ret->x_pos[__x], sizeof(uint), 1, fp);
		FWRITE(&ret->x_t  [__x], sizeof(uint), 1, fp);
	}
	
	//
	FWRITE(&ret->Y, sizeof(uint), 1, fp);
	
	//
	FWRITE(ret->params, sizeof(uint), inst_PARAMS[ret->ID], fp);
};

//	=======================================================

Inst_t * lire_inst_pre_mdl(FILE * fp) {
	Inst_t * ret = lire_tete_instruction(fp);

	//	--- Y & Y' ---
	ret-> y__d = cudalloc<float>(MEGA_T * GRAND_T * ret->Y);
	ret-> l__d = cudalloc<float>(MEGA_T * GRAND_T * ret->L);
	ret->dy__d = cudalloc<float>(MEGA_T * GRAND_T * ret->Y);

	//	--- Poids et Dérivés Locales ---
	ret-> p__d = cudalloc<float>(ret->P);
	ret->dp__d = cudalloc<float>(ret->P);

	//
	init_poids[ret->ID](ret);

	//
	return ret;
};

Inst_t * lire_inst(FILE * fp) {
	Inst_t * ret = lire_tete_instruction(fp);

	float * p = alloc<float>(ret->P);
	FREAD(p, sizeof(float), ret->P, fp);

	//	--- Y & Y' ---
	ret-> y__d = cudalloc<float>(MEGA_T * GRAND_T * ret->Y);
	ret-> l__d = cudalloc<float>(MEGA_T * GRAND_T * ret->L);
	ret->dy__d = cudalloc<float>(MEGA_T * GRAND_T * ret->Y);

	//	--- Poids et Dérivés Locales ---
	ret-> p__d = cpu_vers_gpu<float>(p, ret->P);
	ret->dp__d = cudalloc<float>(ret->P);

	free(p);

	//
	return ret;
};

void ecrire_inst(FILE * fp, Inst_t * inst) {
	ecrire_tete_instruction(fp, inst);
	//
	float * p = gpu_vers_cpu<float>(inst->p__d, inst->P);
	//
	FWRITE(p, sizeof(float), inst->P, fp);
	//
	free(p);
};

void liberer_inst(Inst_t * inst) {
	cudafree<float>(inst-> y__d);
	cudafree<float>(inst-> l__d);
	cudafree<float>(inst->dy__d);
	//
	cudafree<float>(inst-> p__d);
	cudafree<float>(inst->dp__d);
	//
	free(inst);
};

void verif_insts() {
	FOR(0, i, INSTS) {
		ASSERT(inst_Xs[i]     <= MAX_XS);
		ASSERT(inst_PARAMS[i] <= MAX_PARAMS);
	}
};