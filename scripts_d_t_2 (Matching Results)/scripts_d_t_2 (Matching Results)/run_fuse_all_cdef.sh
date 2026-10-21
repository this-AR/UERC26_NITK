#f='E'
#f1='e'
#s='F'
#s1='f'

f='E_sobel_9'
f1='e'
s='F_sobel_9'
s1='f'

c='C'
c1='c'
d='D'
d1='d'

#for C and D
#./max_min_data.sh "../res_"$f1"/LI_"$f11".txt"
#./max_min_data.sh "../res_"$f1"/LM_"$f11".txt"
#./max_min_data.sh "../res_"$f1"/RI_"$f11".txt"
#./max_min_data.sh "../res_"$f1"/RM_"$f11".txt"


#./max_min_data.sh "../res_"$s1"/LI_"$s11".txt"
#./max_min_data.sh "../res_"$s1"/LM_"$s11".txt"
#./max_min_data.sh "../res_"$s1"/RI_"$s11".txt"
#./max_min_data.sh "../res_"$s1"/RM_"$s11".txt"

#for E and F

#./max_min_data.sh "../res_"$f"/LI_"$f1".txt"
#./max_min_data.sh "../res_"$f"/LM_"$f1".txt"
#./max_min_data.sh "../res_"$f"/RI_"$f1".txt"
#./max_min_data.sh "../res_"$f"/RM_"$f1".txt"


#./max_min_data.sh "../res_"$s"/LI_"$s1".txt"
#./max_min_data.sh "../res_"$s"/LM_"$s1".txt"
#./max_min_data.sh "../res_"$s"/RI_"$s1".txt"
#./max_min_data.sh "../res_"$s"/RM_"$s1".txt"

#Score Files
#./fusion_HV.sh "../res_"$f"/LI_"$f1".txtf" "../res_"$s"/LI_"$s1".txtf" "temp_LI_"$f1$s1 LI
#./fusion_HV.sh "../res_"$f"/LM_"$f1".txtf" "../res_"$s"/LM_"$s1".txtf" "temp_LM_"$f1$s1 LM
#./fusion_HV.sh "../res_"$f"/RI_"$f1".txtf" "../res_"$s"/RI_"$s1".txtf" "temp_RI_"$f1$s1 RI
#./fusion_HV.sh "../res_"$f"/RM_"$f1".txtf" "../res_"$s"/RM_"$s1".txtf" "temp_RM_"$f1$s1 RM

#Data Files HV fusion
#./fusion_HV_data.sh "../res_"$c"/LI_"$c1".txt" "../res_"$d"/LI_"$d1".txt" "temp_LI_"$c1$d1 LI
#./fusion_HV_data.sh "../res_"$c"/LM_"$c1".txt" "../res_"$d"/LM_"$d1".txt" "temp_LM_"$c1$d1 LM
#./fusion_HV_data.sh "../res_"$c"/RI_"$c1".txt" "../res_"$d"/RI_"$d1".txt" "temp_RI_"$c1$d1 RI
#./fusion_HV_data.sh "../res_"$c"/RM_"$c1".txt" "../res_"$d"/RM_"$d1".txt" "temp_RM_"$c1$d1 RM

#./fusion_HV_data.sh "../res_"$f"/LI_"$f1".txt" "../res_"$s"/LI_"$s1".txt" "temp_LI_"$f1$s1 LI
#./fusion_HV_data.sh "../res_"$f"/LM_"$f1".txt" "../res_"$s"/LM_"$s1".txt" "temp_LM_"$f1$s1 LM
#./fusion_HV_data.sh "../res_"$f"/RI_"$f1".txt" "../res_"$s"/RI_"$s1".txt" "temp_RI_"$f1$s1 RI
#./fusion_HV_data.sh "../res_"$f"/RM_"$f1".txt" "../res_"$s"/RM_"$s1".txt" "temp_RM_"$f1$s1 RM


#Data Files HV (sgorp and gorp) fusion
./fusion_four.sh "../res_"$c"/LI_"$c1".txt" "../res_"$d"/LI_"$d1".txt"  "../res_"$f"/LI_"$f1".txt" "../res_"$s"/LI_"$s1".txt" "temp_LI_"$c1$d1$f1$s1 LI
./fusion_four.sh "../res_"$c"/LM_"$c1".txt" "../res_"$d"/LM_"$d1".txt"  "../res_"$f"/LM_"$f1".txt" "../res_"$s"/LM_"$s1".txt" "temp_LM_"$c1$d1$f1$s1 LM

./fusion_four.sh "../res_"$c"/RI_"$c1".txt" "../res_"$d"/RI_"$d1".txt"  "../res_"$f"/RI_"$f1".txt" "../res_"$s"/RI_"$s1".txt" "temp_RI_"$c1$d1$f1$s1 RI
./fusion_four.sh "../res_"$c"/RM_"$c1".txt" "../res_"$d"/RM_"$d1".txt"  "../res_"$f"/RM_"$f1".txt" "../res_"$s"/RM_"$s1".txt" "temp_RM_"$c1$d1$f1$s1 RM


#cat "temp_LI_"$c1$d1$f1$s1"_fused/temp_LI_"$c1$d1$f1$s1"_fused.sts" | grep CRR
cat "temp_LI_"$c1$d1$f1$s1"_fused/temp_LI_"$c1$d1$f1$s1"_fused.sts" | grep EER 

#cat "temp_LM_"$c1$d1$f1$s1"_fused/temp_LM_"$c1$d1$f1$s1"_fused.sts" | grep CRR
cat "temp_LM_"$c1$d1$f1$s1"_fused/temp_LM_"$c1$d1$f1$s1"_fused.sts" | grep EER 

#cat "temp_RI_"$c1$d1$f1$s1"_fused/temp_RI_"$c1$d1$f1$s1"_fused.sts" | grep CRR
cat "temp_RI_"$c1$d1$f1$s1"_fused/temp_RI_"$c1$d1$f1$s1"_fused.sts" | grep EER 

#cat "temp_RM_"$c1$d1$f1$s1"_fused/temp_RM_"$c1$d1$f1$s1"_fused.sts" | grep CRR
cat "temp_RM_"$c1$d1$f1$s1"_fused/temp_RM_"$c1$d1$f1$s1"_fused.sts" | grep EER 



#./fusion_HV.sh  temp_LI_cd_fused/temp_LI_cd_fused.dat temp_LI_ef_fused/temp_LI_ef_fused.dat temp_LI_cdef LI
#./fusion_HV.sh  temp_LM_cd_fused/temp_LM_cd_fused.dat temp_LM_ef_fused/temp_LM_ef_fused.dat temp_LM_cdef LM

#./fusion_HV.sh  temp_RI_cd_fused/temp_RI_cd_fused.dat temp_RI_ef_fused/temp_RI_ef_fused.dat temp_RI_cdef RI
#./fusion_HV.sh  temp_RM_cd_fused/temp_RM_cd_fused.dat temp_RM_ef_fused/temp_RM_ef_fused.dat temp_RM_cdef RM


#./fusion_HV.sh  temp_LI_cd_fused/temp_LI_cd_fused.dat ../res/LI_g.txtf temp_LI_cdg LI

