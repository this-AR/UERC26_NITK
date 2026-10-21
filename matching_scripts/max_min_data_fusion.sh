infile=$1

#for fusing codes
#awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" 100000*1*(($7*$7+1000000)/($7*$7*$7*$7*$7))}' $1 > $1"d"

#for individual
#awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" (($7*$7+30)/($7*$7*$7))}' $1 > $1"d"
awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" 1/$7}' $1 > $1"d"

awk '{if ($6>0.0000001) print $0}' $1"d" > $1"a"
max=`awk 'BEGIN {max = -1} {if ($6>max) max=$6} END {print max}' $1"a"`
min=`awk 'BEGIN {min = 100000000000000} {if ($6<min) min=$6} END {print min}' $1"a"`
echo $max $min
range=`calc $max-$min`


awk  -v max=$max -v min=$min '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" ($6-min)/(max-min)}' $1"a" > $1"f1"
awk  '{if($6==0) {print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" 0.0000001} else {print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6}}' $1"f1" > $1"f"

./main.sh $1"f"

#rm -rf $1"a" $1"f1" $1"f"
rm -rf $1"a" $1"f1" 
