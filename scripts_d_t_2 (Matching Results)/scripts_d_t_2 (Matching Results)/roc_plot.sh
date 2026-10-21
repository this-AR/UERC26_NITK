out='ROC3.png'

gnuplot << PLOTTHIS

set term postscript eps enhanced color  "Helvetica" 32 
set terminal png enhanced font 'Verdana,20'
set output "${out}"
set autoscale
unset log                              
unset label
set grid 


set title "ROC based Comparitive Analysis"
set key right top

set logscale x
set yrange [0:100]
# set yrange [0:100]
set xrange [0.01:100]
# set xtics (0.01,0.1,1,5,10,40,100)
# set xtics (1,7,14,28,56,112)

# set xlabel "Row length of input data"
set xlabel "FAR (%) (Plotted in Log-Scale)"
# set xlabel "FAR (%)"
set ylabel "FRR (%)"
# set ylabel "Accuracy (%)"

##############################################################

plot "/home/dxtr/workspace/ego_gait/FPIS_FAR_FRR_Data.dat" using 6:3 title 'FPSI' w linespoints,\
"/home/dxtr/workspace/ego_gait/yedid.dat" using 6:3 title 'EVPR' w linespoints,\
"/home/dxtr/workspace/ego_gait/indoor.dat" using 6:3 title 'DB-01' w linespoints,\
"/home/dxtr/workspace/ego_gait/outdoor.dat" using 6:3 title 'DB-02' w linespoints,\
"/home/dxtr/workspace/ego_gait/combined.dat" using 6:3 title 'DB-03' w linespoints,\
##############################################################
pause mouse
PLOTTHIS