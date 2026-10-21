#f='E'
#f1='e'
#s='F'
#s1='f'

f='E_sobel_9'
f1='e'
s='F_sobel_9'
s1='f'

#f='C'
#f1='c'
#s='D'
#s1='d'


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

#Data Files
#./fusion_HV_data.sh "../res_"$f"/LI_"$f1".txt" "../res_"$s"/LI_"$s1".txt" "temp_LI_"$f1$s1 LI
#./fusion_HV_data.sh "../res_"$f"/LM_"$f1".txt" "../res_"$s"/LM_"$s1".txt" "temp_LM_"$f1$s1 LM
#./fusion_HV_data.sh "../res_"$f"/RI_"$f1".txt" "../res_"$s"/RI_"$s1".txt" "temp_RI_"$f1$s1 RI
#./fusion_HV_data.sh "../res_"$f"/RM_"$f1".txt" "../res_"$s"/RM_"$s1".txt" "temp_RM_"$f1$s1 RM



#./fusion_HV.sh  temp_LI_cd_fused/temp_LI_cd_fused.dat temp_LI_ef_fused/temp_LI_ef_fused.dat temp_LI_cdef LI
#./fusion_HV.sh  temp_LM_cd_fused/temp_LM_cd_fused.dat temp_LM_ef_fused/temp_LM_ef_fused.dat temp_LM_cdef LM

#./fusion_HV.sh  temp_RI_cd_fused/temp_RI_cd_fused.dat temp_RI_ef_fused/temp_RI_ef_fused.dat temp_RI_cdef RI
#./fusion_HV.sh  temp_RM_cd_fused/temp_RM_cd_fused.dat temp_RM_ef_fused/temp_RM_ef_fused.dat temp_RM_cdef RM


#./fusion_HV.sh  temp_LI_cd_fused/temp_LI_cd_fused.dat ../res/LI_g.txtf temp_LI_cdg LI

#for trait in {"LI","LM","RI","RM","FKP"} 

rm -rf log1.txt

for trait in {"LI","LM","RI","RM"} 
do
	for code in {"gorp","sgorp"}
	do
		for grad in {"h","v"}
		do
			./max_min_data.sh "../"$trait"_"$code"_"$grad".txt"
			echo "Working For "$trait"_"$code"_"$grad >> log1.txt
			echo "++++++++++++++++++++++++++++++++++++++" >> log1.txt
			cat $trait"_"$code"_"$grad"/"$trait"_"$code"_"$grad".sts" | grep "(CRR)" >> log1.txt
			cat $trait"_"$code"_"$grad"/"$trait"_"$code"_"$grad".sts" | grep "(EER)" >> log1.txt
			echo "*******************************************************************************" >> log1.txt
		done
	done 	
done

rm -rf log2.txt


for trait in {"LI","LM","RI","RM","FKP"} 
do
			./fusion_four_icvgip.sh "../"$trait"_gorp_v.txt" "../"$trait"_gorp_h.txt" "../"$trait"_sgorp_v.txt" "../"$trait"_sgorp_h.txt" $trait"_all" $trait
			#./max_min_data.sh "../"$trait"_"$code"_"$grad".txt"
			echo "Working For The Fusion of "$trait >> log2.txt
			echo "++++++++++++++++++++++++++++++++++++++" >> log2.txt
			cat $trait"_all_fused/"$trait"_all_fused.sts" | grep "(CRR)" >> log2.txt
			cat $trait"_all_fused/"$trait"_all_fused.sts" | grep "(EER)" >> log2.txt
			echo "*******************************************************************************" >> log2.txt
done

