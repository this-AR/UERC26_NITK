BEGIN {
	#printf "---------------------------------------------------------------------------------------------------\n";
	minI=minG=100;
}
{
CSub=$1;
CPose=$2;
#PSub and CSub are input to the Script initially.
	if(PSub == CSub && PPose == CPose){
		PSub=$1;
		PPose=$2;
		if($5 == 1)
		{
			if(minG>$6)
			{
				minG=$6;argSG=$3;argPG=$4;
			}
		}
		else
		{
			if(minI>$6)
			{
				minI=$6;argSI=$3;argPI=$4;
			}
		}
	}
	else
	{
		#printf "%d %d \t [(%d,%d)%f] \t [(%d,%d)%f]\n", PSub,PPose,argSG,argPG,minG,argSI,argPI,minI;
		printf "%d %d \t [(%d,%d)%f] \t [(%d,%d)%f] \t %f \t %d\n", PSub,PPose,argSG,argPG,minG,argSI,argPI,minI,minG/minI, (minG<=minI);
		minI=minG=100;
		
		##############################
		# Must proceses the read one line		
		PSub=$1;
		PPose=$2;
		if($5 == 1)
		{
			if(minG>$6)
			{
				minG=$6;argSG=$3;argPG=$4;
			}
		}
		else
		{
			if(minI>$6)
			{
				minI=$6;argSI=$3;argPI=$4;
			}
		}
                ##############################		
	}
#printf "%f %d\n", avg/count,count;
}
END {

	printf "%d %d \t [(%d,%d)%f] \t [(%d,%d)%f] \t %f \t %d\n", PSub,PPose,argSG,argPG,minG,argSI,argPI,minI,minG/minI, (minG<=minI);
	#print ARGV[1];
	#college=ARGV[1];
	#printf "College Code/Branch Code: %s || Averages: || %f | %f | %f | %f | %f | %f\n",college,avg1/count,avg2/count,avg3/count,avg4/count,avg5/count,avg6/count;
	#printf "------------------------------------------------------------------------------------------------\n";
}
