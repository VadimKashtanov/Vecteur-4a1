#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define uint unsigned int

#define FREAD(ptr, taille, nb, fp) (void)!fread(ptr, taille, nb, fp);
#define FWRITE(ptr, taille, nb, fp) (void)!fwrite(ptr, taille, nb, fp);
#define FOR(d,i,N) for (uint i=d; i < N; i++)

int main(uint argc, char ** argv) {
	assert(argc == 2);
	//argv[1];
	FILE * fp = fopen("prixs/lignes_dar.bin", "rb");
	//
	uint PRIXS,I,L,N;
	FREAD(&PRIXS, sizeof(uint), 1, fp);
	FREAD(&I,     sizeof(uint), 1, fp);
	FREAD(&L,     sizeof(uint), 1, fp);
	FREAD(&N,     sizeof(uint), 1, fp);
	//
	uint DEPART;
	FREAD(&DEPART, sizeof(uint), 1, fp);
	//
	uint INTERV[I];
	FREAD(INTERV, sizeof(uint), I, fp);
	//
	printf("[\033[96mcree_dar_bin.c\033[0m] Charger PRIXS=%i INTERVS=%i LIGNES=%i N=%i\n", PRIXS,I,L,N);
	//
	float * ligne[I][L];
	FOR(0, i, I) {
		FOR(0, l, L) {
			ligne[i][l] = malloc(sizeof(float) * PRIXS);
			FREAD(ligne[i][l], sizeof(float), PRIXS, fp);
		};
	}
	//
	float * prixs = malloc(sizeof(float) * (PRIXS-DEPART));
	FREAD(  prixs, sizeof(float), (PRIXS-DEPART), fp);
	//
	fclose(fp);
	//
	uint T = PRIXS - DEPART;// - 1;
	//
	printf("[\033[96mcree_dar_bin.c\033[0m] Data_t DEPART=%i à T=%i\n", DEPART, T);
	float * dar = malloc(sizeof(float) * I * T * L * N);
	//
	FOR(0, i, I) {
		FOR(0, t, T) {
			FOR(0, l, L) {
				float vals[N];
				FOR(0, j, N) {
					uint n = N - 1 - j;
					vals[j] = ligne[i][l][DEPART + t - n*INTERV[i]];
				}
				//
				float miu = 0; FOR(0, j, N) miu +=      vals[j]           / (float)N;
				float var = 0; FOR(0, j, N) var += powf(vals[j] - miu, 2) / (float)N;
				//
				FOR(0, j, N) {
					uint n = N - 1 - j;
					//dar[i*(T*L*N) + \
						t*(L*N)   + \
						l*(N)     + \
						n           \
					] = (vals[j] - miu) / sqrtf(var + 1e-8);
					//	.T directement
					dar[i*(T*L*N) + \
						t*(L*N)   + \
						l         + \
						n*L         \
					] = (vals[j] - miu) / sqrtf(var + 1e-8);
				}
			}
		}
	}
	//
	printf("[\033[96mcree_dar_bin.c\033[0m] Ecrire I*T*L*N=%i  %f Mo\n", I*T*L*N, (float)(I*T*L*N)*4 / (float)1e6);
	//
	fp = fopen(argv[1], "wb");
	FWRITE(&I, sizeof(uint), 1, fp);
	FWRITE(&T, sizeof(uint), 1, fp);
	FWRITE(&L, sizeof(uint), 1, fp);
	FWRITE(&N, sizeof(uint), 1, fp);
	FWRITE(  dar, sizeof(float), I*T*L*N, fp);
	FWRITE(prixs, sizeof(float),       T, fp);
	fclose(fp);
	printf("[\033[96mcree_dar_bin.c\033[0m] Ecriture Réussie !\n");

	system("python3 prixs/plumer_un_bloque.py");
};