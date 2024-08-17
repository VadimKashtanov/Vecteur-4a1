#include "btcusdt.cuh"

#include "../../impl_template/tmpl_etc.cu"

BTCUSDT_t * cree_btcusdt(char * fichier) {
	//
	BTCUSDT_t * ret = (BTCUSDT_t*)malloc(sizeof(BTCUSDT_t));

	//
	FILE * fp = FOPEN(fichier, "rb");
	
	//
	FREAD(&ret->I,  sizeof(uint), 1, fp);
	FREAD(&ret->T,  sizeof(uint), 1, fp);
	FREAD(&ret->L,  sizeof(uint), 1, fp);
	FREAD(&ret->N,  sizeof(uint), 1, fp);

	uint I=ret->I;
	uint T=ret->T;
	uint L=ret->L;
	uint N=ret->N;

	ASSERT(ret->I == INTERVS);

	//
	float * serie = lire        <float>(fp,    I*T*L*N);
	ret->serie__d = cpu_vers_gpu<float>(serie, I*T*L*N);
	free(serie);

	//
	float * prixs = lire        <float>(fp,    T);
	ret->prixs__d = cpu_vers_gpu<float>(prixs, T);
	free(prixs);

	//
	fclose(fp);

	ret->X=ret->I*ret->L*ret->N;
	ret->Y=ret->X;

	//
	printf("BTCUSDT : I=%i T=%i L=%i N=%i\n", ret->I, ret->T, ret->L, ret->N);

	//
	return ret;
};

void liberer_btcusdt(BTCUSDT_t * donnee) {
	cudafree<float>(donnee->serie__d);
	cudafree<float>(donnee->prixs__d);
};