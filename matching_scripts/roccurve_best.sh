if [ $# -ne 2 ]; then
	echo "Usage: [1]	Graph Png  File Name"
	echo "       [2]	Base"
	exit 1
fi
################################################################
#FRR
awk '{printf("\n  %8.0f", $1*10000);}'  same.txt > G_N_t.dat
################################################################
#FAR
awk '{printf("\n  %8.0f", $1*10000);}'  different.txt > I_N_t.dat
################################################################


################################################################

#to remove first line
gl=`cat G_N_t.dat | wc -l`
il=`cat I_N_t.dat | wc -l`
#gl=`expr $gl - 1`
#il=`expr $il - 1`

tail -n $gl G_N_t.dat > G_N.dat
tail -n $il I_N_t.dat > I_N.dat

################################################################

./Gcdf.sh -c 1 0 10000 1 5 < G_N.dat > G_cdf.dat
./Icdf.sh -c 1 0 10000 1 5 < I_N.dat > I_cdf.dat 

paste G_cdf.dat I_cdf.dat > "$2"_FAR_FRR_Data_best.dat 
base=`echo "$2" | tr '/' '\n'| tail -1 | cut -d"." -f1`

#######################################################################
#creating FRR vS FAR Data

#######################################################################

#awk '{if($1 > 8500) print $0}'  G_cdf.dat > G_N1.dat
#awk '{if($1 > 8500) print $0}'  I_cdf.dat > I_N1.dat

out=$1
b=$2
t=$base
gnuplot << PLOTTHIS
set term postscript eps enhanced color  "Helvetica" 32

set terminal png
set output "${out}"
set autoscale
unset log                              
unset label
set grid 
set title "FAR Vs FRR"
set key left top
set format x "10^{%L}"
set logscale x
set xtics (0.0001,0.001,0.01,0.1,1,100)
set xlabel "FRR (%)"
set ylabel "FAR (%)"
plot "${b}_FAR_FRR_Data_best.dat" using 3:6 title '${t}' w linespoints
pause mouse
PLOTTHIS
rm -rf G_N.dat I_N.dat G_cdf.dat I_cdf.dat G_N_t.dat I_N_t.dat


