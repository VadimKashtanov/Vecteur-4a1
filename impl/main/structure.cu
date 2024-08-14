#include "main.cuh"

void ecrire_structure_generale(char * file) {
	uint I = 1;
	FILE * fp = FOPEN(file, "wb");
	//
	FWRITE(&I, sizeof(uint), 1, fp);
	//
	uint elements[I] = {
		MEGA_T
	};
	FWRITE(elements, sizeof(uint), I, fp);
	//
	fclose(fp);
};