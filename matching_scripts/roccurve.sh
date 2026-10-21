if [ $# -ne 3 ]; then
	echo "Usage: [1]	Data File"
	echo "       [2]	Graph Png  File Name"
	echo "       [3]	Base"
	exit 1
fi
################################################################
#FRR
awk '{if($5==1) printf("\n  %12.0f", $6*10000);}'  $1 > G_N_t.dat
################################################################
#FAR
awk '{if($5==0) printf("\n  %12.0f", $6*10000);}'  $1 > I_N_t.dat
################################################################

#to remove first line
gl=`cat G_N_t.dat | wc -l`
il=`cat I_N_t.dat | wc -l`
#gl=`expr $gl - 1`
#il=`expr $il - 1`

tail -n $gl G_N_t.dat > G_N.dat
tail -n $il I_N_t.dat > I_N.dat

################################################################

./Gcdf.sh -c 1 0 10000 1 10 < G_N.dat > G_cdf.dat
./Icdf.sh -c 1 0 10000 1 10 < I_N.dat > I_cdf.dat 

paste G_cdf.dat I_cdf.dat > "$3"_FAR_FRR_Data.dat 
base=`echo "$3" | tr '/' '\n'| tail -1 | cut -d"." -f1`

#######################################################################
#creating FRR vS FAR Data

#######################################################################

#awk '{if($1 > 8500) print $0}'  G_cdf.dat > G_N1.dat
#awk '{if($1 > 8500) print $0}'  I_cdf.dat > I_N1.dat

out=$2
b=$3
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

#set ylabel "Frequency (%)"
#set xlabel "Score (%)"
set format x "10^{%L}"

##########################
#set xrange [9100:9500]
#set yrange [0:0.8]
#########################

set logscale x
#set xrange [0.0001:100]
set xtics (0.0001,0.001,0.01,0.1,1,100)

#set xlabel "Threshold"
#set ylabel "FAR/FRR (%)"

set xlabel "FRR (%)"
set ylabel "FAR (%)"

#plot "G_Hist.dat" using 1:2 title 'Genuine' w linespoints, \
#     "I_Hist.dat" using 1:2 title 'Imposter' w linespoints

#plot "G_Hist.dat" using 1:2 title 'Genuine' w points, \
#     "I_Hist.dat" using 1:2 title 'Imposter' w points
##############################################################
#plot "G_cdf.dat" using 1:3 title 'FRR' w linespoints, \
#     "I_cdf.dat" using 1:3 title 'FAR' w linespoints
##############################################################

plot "${b}_FAR_FRR_Data.dat" using 3:6 title '${t}' w linespoints


#plot "G_Hist.dat" using 1:2 title 'Genuine' w points, \
#    "I_Hist.dat" using 1:2 title 'Imposter' w points, \
#    "C_Hist.dat" using 1:2 title 'Overall' w points
pause mouse
PLOTTHIS
rm -rf G_N.dat I_N.dat G_cdf.dat I_cdf.dat


