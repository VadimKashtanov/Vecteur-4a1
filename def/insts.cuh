#pragma once

#include "meta.cuh"
#include "btcusdt.cuh"

#define INSTS 27

#define MAX_PARAMS 6
#define MAX_XS     MAX2(4, INTERVS)

#include "activations_generales.cuh"

typedef struct {
	//
	uint ID;

	//	Parametres
	uint params[MAX_PARAMS];

	//	--- Dar ---
	uint x_Y  [MAX_XS];	//	La taille
	uint x_pos[MAX_XS];	//	L'instruction d'entrée
	uint x_t  [MAX_XS];	//	x[t] 0=0, 1=-1

	//	--- Sorties ---
	uint P;		//	Poids
	uint Y;		//	Sorties
	uint L;		//	Dérivés Locales ou autre

	//	--- Y & Y' ---
	float *  y__d;	//MEGA_T * GRAND_T * Y
	float *  l__d;	//MEGA_T * GRAND_T * L
	float * dy__d;	//MEGA_T * GRAND_T * Y

	//	--- Poids et Dérivés Locales ---
	float *  p__d;	//P
	float * dp__d;	//P

	//
	void * espace_potentiel;
} Inst_t;

//	Initialiser les listes
void init_listes_instructions();

//	Lecture / Ecriture
Inst_t * lire_inst_pre_mdl(FILE * fp);
Inst_t * lire_inst        (FILE * fp);
void     ecrire_inst      (FILE * fp, Inst_t * inst);

//	Mem
void liberer_inst(Inst_t * inst);

//	Zero Y en cas d'inexistance de tous les X
void inst_zero_mega_t(Inst_t * inst, uint mega_t);

//	---------- Listes fonctions ----------

typedef uint (*dimention_f)(uint X[MAX_XS], uint x[MAX_XS], uint t[MAX_XS], uint Y, uint params[MAX_PARAMS]);
typedef void (*inst__f_f  )(Inst_t * inst, float ** x__d,                 uint * ts__d, uint mega_t);
typedef void (*inst_df_f  )(Inst_t * inst, float ** x__d, float ** dx__d, uint * ts__d, uint mega_t);
typedef void (*inst_f     )(Inst_t * inst);

extern uint   inst_Xs    [INSTS];
extern uint   inst_PARAMS[INSTS];
extern char * inst_Nom   [INSTS];
//
extern dimention_f calculer_P[INSTS];
extern dimention_f calculer_L[INSTS];
//
extern inst__f_f __f_inst[INSTS];
extern inst_df_f _df_inst[INSTS];
//
extern inst_f init_poids[INSTS];
//
extern inst_f pre_f[INSTS];

typedef struct {
	uint Xs;
	uint PARAMS;
	char * Nom;
	//
	dimention_f calculer_P;
	dimention_f calculer_L;
	//
	inst_f init_poids;
	//
	inst__f_f  f;
	inst_df_f df;
	//
	inst_f pre_f;
} fonctions_insts_t;

extern fonctions_insts_t fonctions_insts[INSTS];

//	--------------

void verif_insts();