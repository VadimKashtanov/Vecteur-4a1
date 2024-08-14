#include "insts.cuh"

#include "../../impl_template/tmpl_etc.cu"

//	------- Instructions ---------

#include "insts/entree.cuh"
#include "insts/select_vect.cuh"

#include "insts/drop_vecteur.cuh"

#include "insts/activation.cuh"
#include "insts/activation_poid.cuh"

#include "insts/softmax.cuh"

#include "insts/poid.cuh"

#include "insts/matmul.cuh"
#include "insts/matmul_poid_AP.cuh"
#include "insts/matmul_poid_PA.cuh"

#include "insts/QKtDivClef.cuh"

#include "insts/somme.cuh"
#include "insts/sous.cuh"
#include "insts/mul.cuh"
#include "insts/div.cuh"

#include "insts/isomme.cuh"
#include "insts/imaxmin.cuh"

#include "insts/batch_miu.cuh"
#include "insts/batch_variance.cuh"
#include "insts/batch_norm_scale.cuh"

#include "insts/positionnal.cuh"

#include "insts/normalisation.cuh"
#include "insts/div_scal.cuh"

#include "insts/concatenation.cuh"
#include "insts/canalisation.cuh"
#include "insts/union.cuh"
#include "insts/transpose2d.cuh"

//	------------------------------

fonctions_insts_t fonctions_insts[INSTS] = {
	fi_entree,
	fi_select_vect,
	//
	fi_drop_vecteur,
	//
	fi_activation,
	fi_activation_poid,
	//
	fi_softmax,
	//
	fi_poid,
	//
	fi_matmul,
	fi_matmul_poid_AP,
	fi_matmul_poid_PA,
	//
	fi_QKtDivClef,
	//
	fi_somme,
	fi_sous,
	fi_mul,
	fi_div,
	//
	fi_isomme,
	fi_imaxmin,
	//
	fi_batch_miu,
	fi_batch_variance,
	fi_batch_norm_scale,
	//
	fi_positionnal,
	//
	fi_normalisation,
	fi_div_scal,
	//
	fi_concatenation,
	fi_canalisation,
	fi_union,
	fi_transpose2d
};

//	------------------------------

uint inst_Xs    [INSTS] = {};
uint inst_PARAMS[INSTS] = {};
char *  inst_Nom[INSTS] = {};
//
dimention_f calculer_P[INSTS] = {};
dimention_f calculer_L[INSTS] = {};
//
inst__f_f __f_inst[INSTS] = {};
inst_df_f _df_inst[INSTS] = {};
//
inst_f  init_poids[INSTS] = {};
//
inst_f  pre_f[INSTS] = {};

//	-------------------------------

void init_listes_instructions() {
	FOR(0, i, INSTS) {
		inst_Xs    [i] = fonctions_insts[i].Xs;
		inst_PARAMS[i] = fonctions_insts[i].PARAMS;
		inst_Nom   [i] = fonctions_insts[i].Nom;
		//
		calculer_P[i] = fonctions_insts[i].calculer_P;
		calculer_L[i] = fonctions_insts[i].calculer_L;
		//
		__f_inst[i] = fonctions_insts[i].f ;
		_df_inst[i] = fonctions_insts[i].df;
		//
		init_poids[i] = fonctions_insts[i].init_poids;
		//
		pre_f[i] = fonctions_insts[i].pre_f;
	};
};