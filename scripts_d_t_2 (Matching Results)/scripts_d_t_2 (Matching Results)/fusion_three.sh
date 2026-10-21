if [ $# -ne 4 ]; then
	echo "Usage: [1]	File 1"
	echo "Usage: [2]	File 2"
    echo "Usage: [3]	File 3"
	echo "Usage: [4]	Output Folder (base) "
	exit 1
fi
#Do not give a score based result file. Now we are working over data file that contains 7 fields (last two are score and macthing).

base=$4
rm -rf $base"_fused"
mkdir -p $base"_fused"
base=$base"_fused/"$base

##################################################################################################################################################
echo "                                                           "
echo "==================WORKING FOR DATA PROVIDED========================"
echo "                                                           "

paste $1 $2 $3 > $base"_both.dat"
tail $base"_both.dat"

#Since now working over data file it contains 7 fields (last two are score and macthing).
# we fuse them as to make another data file with 7 fields

#here fusion is done
awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" (0.6*$6+0.2*$12+0.2*$18)}' $base"_both.dat" > $base"_fused.dat"

##################################################################################################################################################
#time ./sigmoid_nor.sh ../best_res/LPY100.txt 777 4.5/5
#./main.sh $base"_fused.dat" 1 1 $imgpath
./main.sh $base"_fused.dat"
##################################################################################################################################################
rm -rf $base"_both.dat" 
##################################################################################################################################################








