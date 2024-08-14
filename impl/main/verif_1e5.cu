#include "main.cuh"

#include "../impl_template/tmpl_etc.cu"

/*static void ___test_le_model(BTCUSDT_t * btcusdt, char * fichier) {
	printf(" ======================================================= \n");
	printf(" ============== %s ============== \n", fichier);
	printf(" ======================================================= \n");
	Mdl_t * mdl = cree_mdl_depuis_st_bin(fichier);
	plumer_model(mdl);
	tester_le_model(mdl, btcusdt);
	liberer_mdl(mdl);
};*/

void verif_mdl_1e5() {
	BTCUSDT_t * btcusdt = cree_btcusdt("prixs/dar_8x16.bin");
	//
	ASSERT(btcusdt->X == 128);
	//ASSERT(btcusdt->Y ==   3);
	//
	//ASSERT(GRAND_T == 64);
	//ASSERT(MEGA_T  ==  8);
	//
	//___test_le_model(btcusdt, "mdl_test_inst/activation.st.bin");			//OK
	//___test_le_model(btcusdt, "mdl_test_inst/biais.st.bin");				//OK
	//___test_le_model(btcusdt, "mdl_test_inst/dot1d_X.st.bin");			//OK
	//___test_le_model(btcusdt, "mdl_test_inst/dot1d_XY.st.bin");			//OK
	//___test_le_model(btcusdt, "mdl_test_inst/kconvl1d.st.bin");			//OK
	//___test_le_model(btcusdt, "mdl_test_inst/kconvl1d_stricte.st.bin");	//OK
	//___test_le_model(btcusdt, "mdl_test_inst/kconvl2d_stricte.st.bin");	//OK
	//___test_le_model(btcusdt, "mdl_test_inst/matmul1d.st.bin");			//OK
	//___test_le_model(btcusdt, "mdl_test_inst/matmul1d_canal.st.bin");		//OK
	//___test_le_model(btcusdt, "mdl_test_inst/mul2.st.bin");				//OK
	//___test_le_model(btcusdt, "mdl_test_inst/mul3.st.bin");				//OK
	//___test_le_model(btcusdt, "mdl_test_inst/pool1d.st.bin");			//OK
	//___test_le_model(btcusdt, "mdl_test_inst/pool2x2_2d.st.bin");         //OK
	//                                     14                               //!!!!!
	//___test_le_model(btcusdt, "mdl_test_inst/somme2.st.bin");				//OK
	//___test_le_model(btcusdt, "mdl_test_inst/somme3.st.bin");				//OK
	//___test_le_model(btcusdt, "mdl_test_inst/somme4.st.bin");				//OK
	//___test_le_model(btcusdt, "mdl_test_inst/Y.st.bin");					//OK
	//___test_le_model(btcusdt, "mdl_test_inst/Y_canalisation.st.bin");		//OK
	//
	liberer_btcusdt(btcusdt);
};