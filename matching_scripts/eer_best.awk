BEGIN {
c=0;PP=pose;MIN=1000;
}
{
diff=$3-$6;
if(diff<0)
{
diff=diff*(-1);
}

if(diff < MIN)
{
MIN=diff;
TH=$1;
eer=($3+$6)/2;
#print $0
#exit
}
}
END {
printf "Equal Error Rate         (Best EER) = \t (%f)  %f with Difference = %f\n",eer,TH/10000,MIN;
	
}
