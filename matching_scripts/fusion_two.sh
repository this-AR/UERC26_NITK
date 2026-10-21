if [ $# -ne 4 ]; then
	echo "Usage: [1]	Hor Data File"
	echo "Usage: [2]	Ver Data File"
	echo "Usage: [3]	Output Folder (base) "
	echo "Usage: [4]	Trait ID (LI,LM,RI,RM) "
	exit 1
fi

base=$3
rm -rf $base"_fused"
mkdir -p $base"_fused"
base=$base"_fused/"$base

imgpath="/db/"$4
##################################################################################################################################################
echo "                                                           "
echo "==================WORKING FOR DATA PROVIDED========================"
echo "                                                           "

paste $1 $2 > $base"_both.dat"

awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $12}' $base"_both.dat" > $base"_allinone.dat"

awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" (0.85*$6)+(0.15*$7) }' $base"_allinone.dat" > $base"_fused.dat"
##################################################################################################################################################
#time ./sigmoid_nor.sh ../best_res/LPY100.txt 777 4.5/5
#./main.sh $base"_fused.dat" 1 1 $imgpath
./main.sh $base"_fused.dat"
##################################################################################################################################################
rm -rf $base"_both.dat" $base"_allinone.dat"
##################################################################################################################################################










