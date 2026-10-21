if [ $# -ne 2 ]; then
	echo "Usage: [1]	Data File"
	echo "       [2]	Graph Png File Name"
	exit 1
fi
out=$2
gnuplot << PLOTTHIS
#set term postscript eps enhanced color
set terminal png
set output "${out}"
set autoscale
unset log                              
unset label
set grid 
set title "Genuine Score Vs Imposter Score Frequency Histogram"

set ylabel "Frequency (%)"
set xlabel "Score (%)"
#set format x "10^{%L}"
set xrange [0:1000]
#set yrange [0:100]
#set logscale x
#set xrange [0:1]
#set xtics (0.0001,0.001,0.01,0.1,0,1)
#set xlabel "Threshold"
#set ylabel "FAR/FRR (%)"
#plot "$1" using 1:2 title 'FRR' w points, \
#	"$1" using 1:3 title 'FAR' w points
#plot [0.4:0.6] "$1" using 1:2 title 'FRR' w linespoints, \
#	"$1" using 1:3 title 'FAR' w linespoints

plot "G_Hist.dat" using 1:2 title 'Genuinea' w points, \
     "I_Hist.dat" using 1:2 title 'Imposter' w points
pause mouse
PLOTTHIS
