#include "mdl.cuh"

static void monter_de_un(uint I, uint i, uint * grille, uint * nb) {
	FOR(i+1, j, I) {
		FOR(0, k, nb[j]) {
			grille[(j-1)*I + k] = grille[j*I + k];
		};
		nb[j-1] = nb[j];
		nb[j] = 0;
	};
};

static void eliminer_vides(uint I, uint * grille, uint * nb) {
	uint des_trous;
	do {
		des_trous = false;
		uint pos_nulle = I+1;
		FOR(0, i, I) {
			uint des_choses_apres = false;
			FOR(i, j, I) if (nb[j] != 0) des_choses_apres = true;
			if (nb[i] == 0 && des_choses_apres) {
				pos_nulle = i;
				des_trous = true;
			}
		}
		//
		if (pos_nulle != I+1) {
			monter_de_un(I, pos_nulle, grille, nb);
		}
	} while (des_trous);
};

//	===============================================================

static void deplacer(
	uint de_ligne, uint vers_ligne, uint de_elm,
	uint I, uint * grille, uint * nb)
{
	nb[vers_ligne] += 1;
	grille[vers_ligne*I + nb[vers_ligne]-1] = grille[de_ligne*I + de_elm];
	//
	FOR(de_elm+1, j, nb[de_ligne]) grille[de_ligne*I + j-1] = grille[de_ligne*I + j];
	//
	nb[de_ligne] -= 1;
	//
	eliminer_vides(I, grille, nb);
};

//	===============================================================

static void mise_a_jour_position(
	uint * positions_ligne, uint * position_elm,
	uint I, uint * grille, uint * nb)
{
	FOR(0, i, I) {
		uint fait = false;
		FOR(0, j, I) {
			if (nb[j] != 0)  {
				FOR(0, k, nb[j]) {
					if (grille[j*I + k] == i) {
						positions_ligne[i] = j;
						position_elm   [i] = k;
						//
						fait = true;
						break;
					}
				}
			}
			if (fait) break;
		}
	}
};

void mdl_optimisation(Mdl_t * mdl) {
	uint I = mdl->insts;
	//
	uint grille[I*I];
	uint     nb[I];
	FOR(0, i, I) {
		grille[i*I + 0] = i;
		nb[i]        = 1;
	}
	//
	uint positions_ligne[I];
	uint position_elm   [I];
	mise_a_jour_position(positions_ligne, position_elm, I, grille, nb);
	//
	uint optimisable = false;
	do {
		//
		/*printf(" ---------- Grille --------------\n");
		FOR(0, i, I) {
			printf("%i| ", i);
			FOR(0, j, nb[i]) printf("%i ", grille[i*I + j]);
			printf("\n");
		}*/
		//
		optimisable = false;
		//
		ASSERT(mdl->inst[0]->ID == 0);
		FOR(1, i, I) {//1 car i_Entree n'est pas optimisable
			//# Si entrée (xt!=1) de l'inst est dans la ligne au dessus -> OK. Peut pas faire mieux.
			//# Sinon -> Optimisable=True & Monter l'inst d'une ligne. Et Break

			uint max_entree = 0;
			FOR(0, j, inst_Xs[mdl->inst[i]->ID]) {
				if (mdl->inst[i]->x_t[j] != 1) {
					uint où = positions_ligne[mdl->inst[i]->x_pos[j]];
					max_entree = (max_entree > où ? max_entree : où);
				}
			};
			//
			if (max_entree == positions_ligne[i] - 1) {
				optimisable = false;
			} else if (max_entree > positions_ligne[i] - 1) {
				printf("inst=%i| max_entree=%i positions_ligne[i]-1=%i\n", i, max_entree, positions_ligne[i] - 1);
				ERR("Il y a une couille dans le paté. max_entree > positions_ligne[i] - 1");
			} else {
				optimisable = true;
				//
				uint de_ligne = positions_ligne[i];
				uint vers_ligne = positions_ligne[i] - 1;
				uint de_elm = position_elm[i];
				//
				deplacer(
					de_ligne, vers_ligne, de_elm,
					I, grille, nb);
				//
				mise_a_jour_position(positions_ligne, position_elm, I, grille, nb);
				//
				break;
			}
			//
			if (optimisable) break;
		}
	} while (false && optimisable);
	MSG("Pas d'optimisation car trop lent")
	//
	/*printf(" ---------- Grille FINALE --------------\n");
	FOR(0, i, I) {
		printf("%i| ", i);
		FOR(0, j, nb[i]) printf("%i ", grille[i*I + j]);
		printf("\n");
	}*/
	//
	uint premier_nulle = mdl->insts;
	FOR(0, i, I) {
		if (nb[i] == 0) {
			premier_nulle = i;
			break;
		}
	}
	//
	mdl->BLOQUES = premier_nulle-1+1;
	//
	mdl->elements = (uint*)malloc(sizeof(uint) * mdl->BLOQUES);
	FOR(0, i, mdl->BLOQUES)
		mdl->elements[i] = nb[i];
	//
	mdl->instructions = (uint**)malloc(sizeof(uint*) * mdl->BLOQUES);
	FOR(0, i, mdl->BLOQUES) {
		mdl->instructions[i] = (uint*)malloc(sizeof(uint) * nb[i]);
		FOR(0, j, nb[i])
			mdl->instructions[i][j] = grille[i*I + j];
	}
};