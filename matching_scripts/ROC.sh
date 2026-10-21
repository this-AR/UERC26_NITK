
out='CASIA-Angle.png'

gnuplot << PLOTTHIS
set term postscript eps enhanced color  "Helvetica" 32

set terminal png
set output "${out}"
set autoscale
unset log                              
unset label
set grid 

set title "ROC based Comparative Analysis"
set key right top

set logscale x
set xrange [0.01:10]
set yrange [0:62]
#set yrange [0.01:100]
#set xtics (0.01,0.1,1,5,10,40,100)

set xlabel "FAR (%) (Plotted in Log-Scale)"
set ylabel "FRR (%)"

##############################################################

plot "a1Data.dat" using 6:3 title '0-degree' w linespoints,\
"a2.dat" using 6:3 title '90-degree' w linespoints
##############################################################
pause mouse
PLOTTHIS

