#valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./main
#cuda-gdb
#compute-sanitizer --tool memcheck
rm *.o
#clear
printf "[\033[93m***\033[0m] \033[103mCompilation ...\033[0m \n"

echo "!!! -G -g !!!";A="-Idef -Idef/insts -diag-suppress 2464 -G -g -O0 -lm -lcublas_static -lcublasLt_static -lculibos -Xcompiler -fopenmp -Xcompiler -O0"
#echo "!!! -g !!!";A="-Idef -Idef/insts -diag-suppress 2464 -g -O0 -lm -lcublas_static -lcublasLt_static -lculibos -Xcompiler -fopenmp -Xcompiler -O3"
#A="-Idef -Idef/insts -diag-suppress 2464 -O3 -lm -lcublas_static -lcublasLt_static -lculibos -Xcompiler -fopenmp -Xcompiler -O3"

# les 3 lignes au dessus : debbug cuda, debbug, optimiser

################################################################

#	/insts
nvcc -c impl/insts/insts.cu         ${A} &
nvcc -c impl/insts/insts_io.cu      ${A} &
nvcc -c impl/insts/inst_controle.cu ${A} &
#	/insts/_entree
nvcc -c impl/insts/entree/entree.cu ${A} &
nvcc -c impl/insts/entree/entree_f.cu ${A} &
nvcc -c impl/insts/entree/entree_df.cu ${A} &
#	/insts/select_vect
nvcc -c impl/insts/select_vect/select_vect.cu ${A} &
nvcc -c impl/insts/select_vect/select_vect_f.cu ${A} &
nvcc -c impl/insts/select_vect/select_vect_df.cu ${A} &
#	/insts/drop_vecteur
nvcc -c impl/insts/drop_vecteur/drop_vecteur.cu ${A} &
nvcc -c impl/insts/drop_vecteur/drop_vecteur_f.cu ${A} &
nvcc -c impl/insts/drop_vecteur/drop_vecteur_df.cu ${A} &
#	/insts/activation
nvcc -c impl/insts/activation/activation.cu ${A} &
nvcc -c impl/insts/activation/activation_f.cu ${A} &
nvcc -c impl/insts/activation/activation_df.cu ${A} &
#	/insts/activation_poid
nvcc -c impl/insts/activation_poid/activation_poid.cu ${A} &
nvcc -c impl/insts/activation_poid/activation_poid_f.cu ${A} &
nvcc -c impl/insts/activation_poid/activation_poid_df.cu ${A} &
#	/insts/softmax
nvcc -c impl/insts/softmax/softmax.cu ${A} &
nvcc -c impl/insts/softmax/softmax_f.cu ${A} &
nvcc -c impl/insts/softmax/softmax_df.cu ${A} &
#	/insts/poid
nvcc -c impl/insts/poid/poid.cu ${A} &
nvcc -c impl/insts/poid/poid_f.cu ${A} &
nvcc -c impl/insts/poid/poid_df.cu ${A} &
#	/insts/matmul
nvcc -c impl/insts/matmul/matmul.cu ${A} &
nvcc -c impl/insts/matmul/matmul_f.cu ${A} &
nvcc -c impl/insts/matmul/matmul_df.cu ${A} &
#	/insts/matmul_poid_AP
nvcc -c impl/insts/matmul_poid_AP/matmul_poid_AP.cu ${A} &
nvcc -c impl/insts/matmul_poid_AP/matmul_poid_AP_f.cu ${A} &
nvcc -c impl/insts/matmul_poid_AP/matmul_poid_AP_df.cu ${A} &
#	/insts/matmul_poid_PA
nvcc -c impl/insts/matmul_poid_PA/matmul_poid_PA.cu ${A} &
nvcc -c impl/insts/matmul_poid_PA/matmul_poid_PA_f.cu ${A} &
nvcc -c impl/insts/matmul_poid_PA/matmul_poid_PA_df.cu ${A} &
#	/insts/QKtDivClef
nvcc -c impl/insts/QKtDivClef/QKtDivClef.cu ${A} &
nvcc -c impl/insts/QKtDivClef/QKtDivClef_f.cu ${A} &
nvcc -c impl/insts/QKtDivClef/QKtDivClef_df.cu ${A} &
#	/insts/somme
nvcc -c impl/insts/somme/somme.cu ${A} &
nvcc -c impl/insts/somme/somme_f.cu ${A} &
nvcc -c impl/insts/somme/somme_df.cu ${A} &
#
wait
#	/insts/sous
nvcc -c impl/insts/sous/sous.cu ${A} &
nvcc -c impl/insts/sous/sous_f.cu ${A} &
nvcc -c impl/insts/sous/sous_df.cu ${A} &
#	/insts/mul
nvcc -c impl/insts/mul/mul.cu ${A} &
nvcc -c impl/insts/mul/mul_f.cu ${A} &
nvcc -c impl/insts/mul/mul_df.cu ${A} &
#	/insts/div
nvcc -c impl/insts/div/div.cu ${A} &
nvcc -c impl/insts/div/div_f.cu ${A} &
nvcc -c impl/insts/div/div_df.cu ${A} &
#	/insts/isomme
nvcc -c impl/insts/isomme/isomme.cu ${A} &
nvcc -c impl/insts/isomme/isomme_f.cu ${A} &
nvcc -c impl/insts/isomme/isomme_df.cu ${A} &
#	/insts/imaxmin
nvcc -c impl/insts/imaxmin/imaxmin.cu ${A} &
nvcc -c impl/insts/imaxmin/imaxmin_f.cu ${A} &
nvcc -c impl/insts/imaxmin/imaxmin_df.cu ${A} &
#	/insts/batch_miu
nvcc -c impl/insts/batch_miu/batch_miu.cu ${A} &
nvcc -c impl/insts/batch_miu/batch_miu_f.cu ${A} &
nvcc -c impl/insts/batch_miu/batch_miu_df.cu ${A} &
#	/insts/batch_variance
nvcc -c impl/insts/batch_variance/batch_variance.cu ${A} &
nvcc -c impl/insts/batch_variance/batch_variance_f.cu ${A} &
nvcc -c impl/insts/batch_variance/batch_variance_df.cu ${A} &
#	/insts/batch_norm_scale
nvcc -c impl/insts/batch_norm_scale/batch_norm_scale.cu ${A} &
nvcc -c impl/insts/batch_norm_scale/batch_norm_scale_f.cu ${A} &
nvcc -c impl/insts/batch_norm_scale/batch_norm_scale_df.cu ${A} &
#	/insts/positionnal
nvcc -c impl/insts/positionnal/positionnal.cu ${A} &
nvcc -c impl/insts/positionnal/positionnal_f.cu ${A} &
nvcc -c impl/insts/positionnal/positionnal_df.cu ${A} &
#	/insts/div_scal
nvcc -c impl/insts/div_scal/div_scal.cu ${A} &
nvcc -c impl/insts/div_scal/div_scal_f.cu ${A} &
nvcc -c impl/insts/div_scal/div_scal_df.cu ${A} &
#	/insts/normalisation
nvcc -c impl/insts/normalisation/normalisation.cu ${A} &
nvcc -c impl/insts/normalisation/normalisation_f.cu ${A} &
nvcc -c impl/insts/normalisation/normalisation_df.cu ${A} &
#	/insts/concatenation
nvcc -c impl/insts/concatenation/concatenation.cu ${A} &
nvcc -c impl/insts/concatenation/concatenation_f.cu ${A} &
nvcc -c impl/insts/concatenation/concatenation_df.cu ${A} &
#	/insts/canalisation
nvcc -c impl/insts/canalisation/canalisation.cu ${A} &
nvcc -c impl/insts/canalisation/canalisation_f.cu ${A} &
nvcc -c impl/insts/canalisation/canalisation_df.cu ${A} &
#	/insts/union
nvcc -c impl/insts/union/union.cu ${A} &
nvcc -c impl/insts/union/union_f.cu ${A} &
nvcc -c impl/insts/union/union_df.cu ${A} &
#	/insts/transpose2d
nvcc -c impl/insts/transpose2d/transpose2d.cu ${A} &
nvcc -c impl/insts/transpose2d/transpose2d_f.cu ${A} &
nvcc -c impl/insts/transpose2d/transpose2d_df.cu ${A} &
#
wait
###########
#	/etc
nvcc -c impl/etc/etc.cu     ${A} &
#	/etc/btcusdt
nvcc -c impl/btcusdt/btcusdt.cu          ${A} &
nvcc -c impl/btcusdt/btcusdt_f_df.cu        ${A} &
nvcc -c impl/btcusdt/btcusdt_pourcent.cu ${A} &
#	/main
nvcc -c impl/main/verif_1e5.cu    ${A} &
nvcc -c impl/main/structure.cu    ${A} &
#	/mdl
nvcc -c impl/mdl/mdl.cu ${A} &
nvcc -c impl/mdl/mdl_optimisation.cu    ${A} &
nvcc -c impl/mdl/mdl_ctrl.cu            ${A} &
nvcc -c impl/mdl/mdl_f.cu  ${A} &
nvcc -c impl/mdl/mdl_df.cu ${A} &
nvcc -c impl/mdl/mdl_S.cu            ${A} &
nvcc -c impl/mdl/mdl_allez_retour.cu ${A} &
nvcc -c impl/mdl/mdl_pourcent.cu ${A} &
nvcc -c impl/mdl/mdl_tester_le_model.cu ${A} &
nvcc -c impl/mdl/mdl_plume.cu           ${A} &
#	/opti
nvcc -c impl/opti/opti.cu ${A} &
#
nvcc -c impl/opti/opti_sgd.cu     ${A} &
nvcc -c impl/opti/opti_moment.cu  ${A} &
nvcc -c impl/opti/opti_rmsprop.cu ${A} &
nvcc -c impl/opti/opti_adam.cu    ${A} &
#
#	Attente de terminaison des differents fils de compilation
#
wait

################################################################

#	Programme : "principale"
nvcc     -c impl/main.cu ${A}
nvcc *.o -o      main    ${A}
rm main.o

#	Programme : "tester le model"
nvcc     -c impl/prog_tester_le_mdl.cu ${A}
nvcc *.o -o      prog_tester_le_mdl    ${A}
rm prog_tester_le_mdl.o

#	Programme : "prog_verif_f_df_insts"
#nvcc     -c impl/prog_verif_f_df_insts.cu ${A}
#nvcc *.o -o      prog_verif_f_df_insts    ${A}
#rm prog_verif_f_df_insts.o
#
#	Attente de terminaison des differents fils de compilation
#
wait

rm *.o

################################################################

#	Verification d'erreure
if [ $? -eq 1 ]
then
	printf "\n[\033[91m***\033[0m] \033[101m [!] Erreure. Pas d'execution. [!]\033[0m\n"
	exit
else
	printf "[\033[92m***\033[0m] \033[102m========= Execution du programme =========\033[0m\n"
fi

#	Executer si Aucune erreur
time ./main
if [ $? -ne 0 ]
then
	printf "[\033[91m***\033[0m] \033[101m /!\ Erreur durant l'execution. /!\ \033[0m\n"
	exit
else
	printf "[\033[92m***\033[0m] \033[102mAucune erreure durant l'execution.\033[0m\n"
fi