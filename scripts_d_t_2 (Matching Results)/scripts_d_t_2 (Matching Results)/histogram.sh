if [ $# -ne 4 ]; then
	echo "Usage: [1]	Data File"
	echo "       [2]	Graph Png  File Name"
	echo "       [3]	Statistics File Name"
	echo "       [4]	Base"
	exit 1
fi

#echo "Overall Truncation Going On"
#awk '{printf("\n  %8.0f", $6*1000);}'  $1 > C_N.dat
#echo "Overall Truncation Done"

echo "Genuine Truncation Going On"
awk '{if($5==1) printf("\n  %12.0f", $6*10000);}'  $1 > G_N.dat
echo "Genuine  Truncation Done"

echo "Imposter Truncation Going On"
awk '{if($5==0) printf("\n  %12.0f", $6*10000);}'  $1 > I_N.dat
echo "Imposter Truncation Done"

#echo "Overall Histogram Formation Going"
#awk ' NF > 0{ counts[$0] = counts[$0] + 1; } END { for (word in counts) print word, counts[word]; }' C_N.dat > C_N_H.dat
#echo "Overall  Histogram Formation Done"
#echo "Overall  Histogram Sorting Going on"
#sort C_N_H.dat > C_H.dat
#echo "Overall  Histogram Sorting Done"


echo "Genuine  Histogram Formation Going"
awk ' NF > 0{ counts[$0] = counts[$0] + 1; } END { for (word in counts) print word, counts[word]; }' G_N.dat > G_N_H.dat
echo "Genuine  Histogram Formation Done"
echo "Genuine  Histogram Sorting Going on"
sort G_N_H.dat > G_H.dat
echo "Genuine  Histogram Sorting Done"

echo "Imposter Histogram Formation Going"
awk ' NF > 0{ counts[$0] = counts[$0] + 1; } END { for (word in counts) print word, counts[word]; }' I_N.dat > I_N_H.dat
echo "Imposter Histogram Formation Done"
echo "Imposter  Histogram Sorting Goingon"
sort I_N_H.dat > I_H.dat
echo "Genuine  Histogram Sorting Done"

#C_Total=`cat C_N.dat| wc -l`
G_Total=`cat G_N.dat| wc -l`
I_Total=`cat I_N.dat| wc -l`

#awk -v Total=$C_Total '{printf(" %d \t %8.6f \n", $1,($2/Total) * 100);}'  C_H.dat > C_Hist.dat 
awk -v Total=$G_Total '{printf(" %d \t %12.6f \n", $1,($2/Total) * 100);}'  G_H.dat > "$4"_G_Hist.dat
awk -v Total=$I_Total '{printf(" %d \t %12.6f \n", $1,($2/Total) * 100);}'  I_H.dat > "$4"_I_Hist.dat

###################################################################
out=$2
b=$4
gnuplot << PLOTTHIS
#set term postscript eps enhanced color
	

#set style fill transparent solid 1 noborder

set terminal png
set output "${out}"
set autoscale
unset log                              
unset label
set grid 
set title "Genuine Score Vs Imposter Score Frequency Histogram"
set key left top
set ylabel "Frequency (%)"
set xlabel "Score (%)"
#set format x "10^{%L}"
set xrange [0:10001]
#set yrange [0:20]
#set logscale x
#set xrange [0:1]
#set xtics (0.0001,0.001,0.01,0.1,0,1)
#set xlabel "Threshold"
#set ylabel "FAR/FRR (%)"

#plot "G_Hist.dat" using 1:2 title 'Genuine' w linespoints, \
#     "I_Hist.dat" using 1:2 title 'Imposter' w linespoints

#plot "G_Hist.dat" using 1:2 title 'Genuine' w points, \
#     "I_Hist.dat" using 1:2 title 'Imposter' w points

plot "${b}_G_Hist.dat" using 1:2 title 'Genuine' w impulses, \
     "${b}_I_Hist.dat" using 1:2 title 'Imposter' w impulses

#plot "${b}_G_Hist.dat" using 1:2 title 'Genuine' w points, \
#     "${b}_I_Hist.dat" using 1:2 title 'Imposter' w points


#plot "G_Hist.dat" using 1:2 title 'Genuine' w points, \
#    "I_Hist.dat" using 1:2 title 'Imposter' w points, \
#    "C_Hist.dat" using 1:2 title 'Overall' w points
pause mouse
PLOTTHIS
#######################################################################
echo "**************************************************************************************************************************************" > $3
echo "Generating Statistics"
echo "Genuine" >> $3
echo "===========================" >> $3
awk '{print $2 "\t" $1}' "$4"_G_Hist.dat > temp.n
awk -f stat.awk temp.n >> $3

#################################
mu1="`awk -f stat.awk temp.n | grep "average" | awk '{print $1}'`"
sigma1=`awk -f stat.awk temp.n | grep "deviation" | awk '{print $1}'`
#################################

echo "**************************************************************************************************************************************" >> $3
echo "Imposter" >> $3
echo "===========================" >> $3
awk '{print $2 "\t" $1}' "$4"_I_Hist.dat > temp.n
awk -f stat.awk temp.n >> $3

#################################
mu2=`awk -f stat.awk temp.n | grep "average" | awk '{print $1}'`
sigma2=`awk -f stat.awk temp.n | grep "deviation" | awk '{print $1}'`
#################################
################################
#Calculating d'
#echo $mu1 $mu2 $sigma1 $sigma2
N=`calc $mu1-$mu2`;
D1=`calc $sigma1*$sigma1`;
D2=`calc $sigma2*$sigma2`;
D3=`calc $D1+$D2`;
D4=`calc $D3/2`;
D5=`calc "$D4^0.5"`;
d=`calc "$N/$D5"`;
#d=` expr (($mu1-$mu2)/sqrt(($sigma1 * $sigma1)+($sigma2*$sigma2)/2))`;
#d=` expr (($mu1-$mu2)/sqrt(($sigma1 * $sigma1)+($sigma2*$sigma2)/2))`;
echo "**************************************************************************************************************************************" >> $3
echo "Performance Parameters" >> $3
echo "===========================" >> $3
echo "Decidability Index        (DI) = $d" >> $3
################################

echo "Removing Temporary Files"
rm -rf C_N.dat C_N_H.dat G_N_H.dat I_N_H.dat C_H.dat G_H.dat I_H.dat C_Hist.dat G_Hist.dat I_Hist.dat temp.n
#rm -rf C_N.dat G_N.dat I_N.dat C_N_H.dat G_N_H.dat I_N_H.dat C_H.dat G_H.dat I_H.dat C_Hist.dat G_Hist.dat I_Hist.dat temp.n



