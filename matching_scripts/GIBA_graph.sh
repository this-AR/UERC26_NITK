out=$1
base=$2
gnuplot << PLOTTHIS
set term postscript eps enhanced color  "Helvetica" 32

set terminal png
set output "${out}"
set autoscale
unset log                              
unset label
set grid 
set title "Genuine Vs Imposter Best Matching Score Separation Graph"
set key bottom left

#set ylabel "Frequency (%)"
#set xlabel "Score (%)"
#set format x "10^{%L}"
#set format y "10^{%L}"

##########################
#set xrange [0:0.0001]
#set yrange [0.85:1]
#########################

#set logscale x
#set logscale y
#set xrange [10:100]
#set xtics (0.01,0.01,0.1)

#set xlabel "Threshold"
#set ylabel "FAR/FRR (%)"

set xlabel "Test Image"
set ylabel "Normalized Score"


plot "$base" using 7 title 'Best Genuine Score' w linespoints, \
	"$base" using 9 title 'Best Imposter Score' w linespoints

#, \
#	"$base" using 10 title 'Average Imposter Score' w linespoints






pause mouse
PLOTTHIS


