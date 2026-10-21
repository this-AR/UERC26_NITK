if [ $# -ne 6 ]; then
	echo "Usage: [1]	Ver Data File gorp"
	echo "Usage: [2]	Hor Data File gorp"
    echo "Usage: [3]	Ver Data File sgorp"
	echo "Usage: [4]	Hor Data File sgorp"
	echo "Usage: [5]	Output Folder (base) "
	echo "Usage: [6]	Trait ID (LI,LM,RI,RM) "
	exit 1
fi
#Do not give a score based result file. Now we are working over data file that contains 7 fields (last two are score and macthing).

base=$5
rm -rf $base"_fused"
mkdir -p $base"_fused"
base=$base"_fused/"$base

imgpath="/media/naditya/data11/database/Knuckle/db/"$6
##################################################################################################################################################
echo "                                                           "
echo "==================WORKING FOR DATA PROVIDED========================"
echo "                                                           "

paste $1 $2 $3 $4 > $base"_both.dat"
tail $base"_both.dat"
#Since now working over data file it contains 7 fields (last two are score and macthing).
# we fuse them as to make another data file with 7 fields

#here fusion is done
awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" (0.45*$6+0.3*$13+0.2*$20+0.05*$27)"\t" (0.4*$7+0.35*$14+0.25*$21+0.0*$28)}' $base"_both.dat" > $base"_fused.dat"

#but file is not a result file but it is a data file

#Hence run data max min script.

./max_min_data.sh $base"_fused.dat"

#awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $12}' $base"_both.dat" > $base"_allinone.dat"

#awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" (0.6*$6)+(0.4*$7) }' $base"_allinone.dat" > $base"_fused.dat"
##################################################################################################################################################
#time ./sigmoid_nor.sh ../best_res/LPY100.txt 777 4.5/5
#./main.sh $base"_fused.dat" 1 1 $imgpath
#./main.sh $base"_fused.dat"
##################################################################################################################################################
#rm -rf $base"_both.dat" $base"_allinone.dat"
#rm -rf $base"_both.dat"
##################################################################################################################################################










