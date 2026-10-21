if [ $# -ne 3 ]; then
	echo "Usage: [1]	Input Data File"
	echo "       [2]	Threshold"
	echo "       [3]	Genuine(1) else Imposter"
	exit 1
fi

if [ $3 -ne 1 ]; then
echo "Imposter Passed (FAR)"
awk -v th=$2 '{if($5==0 && $6 < th) print $0}'  $1 
echo "FAR Cases Done"
else
echo "Genuine Failed (FRR)"
awk -v th=$2 '{if($5==1 && $6 > th) print $0}'  $1 
echo "FRR Cases Done"
fi














