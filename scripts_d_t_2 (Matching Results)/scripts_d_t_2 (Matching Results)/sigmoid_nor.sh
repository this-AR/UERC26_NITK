if [ $# -ne 3 ]; then
	echo "Usage: [1]	Input Data File"
	echo "       [2]	MU"
	echo "       [3]	Tu"
	exit 1
fi
base=`echo "$1" | tr '/' '\n'| tail -1 | cut -d"." -f1`
awk -v t=$3,mu=$2 '{$6=(1/(1+exp(-t*($6-mu))));print $0 }' $1 

