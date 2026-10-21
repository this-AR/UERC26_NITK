TotalFR=`awk '{if($1==$3){print $0}}' $1  | awk -v th=$2 '{if($6 >= th){print $0}}' | wc -l`
TotalG=`awk '{if($1==$3){print $0}}' $1  | wc -l`
FRR=`calc 100*$TotalFR/$TotalG`
TotalFA=`awk '{if($1!=$3){print $0}}' $1  |awk -v th=$2 '{if($6 < th){print $0}}' | wc -l`
TotalI=`awk '{if($1!=$3){print $0}}' $1  | wc -l`
FRR=`calc 100*$TotalFR/$TotalG`
FAR=`calc 100*$TotalFA/$TotalI`
#echo "Threshlod = " $2 "FRR = " $FRR "FAR = " $FAR
echo $2 "	" $FRR "	" $FAR
