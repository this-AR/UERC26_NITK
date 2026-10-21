# Developed by ADITYA NIGAM
#Phd Student
#IIT KANPUR.

if [ $# -ne 1 ]; then
clear
	echo "Usage: [1]	Input Data File"
echo "For Test Run"
echo "./main.sh test.txt"
echo "			"
echo "Input Data File Format Expected : "
echo "Must contain total 6 fields (TAB seperated) as given in the test.txt file."
echo "[1] 1st Field :	Subject id of first image to be matched"
echo "[2] 2nd Field :	Pose id of first image to be matched"
echo "[3] 3rd Field :	Subject id of second image to be matched"
echo "[4] 4th Field :	Pose id of second image to be matched"
echo "[5] 5th Field :	Binary Field. If Genuine then 1 else 0"
echo "[6] 6th Field :	Normalized Matching Score (Must lie b/w 0 to 1)"
echo "*****************************"
echo "Developed by ADITYA NIGAM"
echo "Ph.D Student"
echo "IIT KANPUR"
echo "Email: naditya@cse.iitk.ac.in"
echo "*****************************"
	exit 1
fi


base=`echo "$1" | tr '/' '\n'| tail -1 | cut -d"." -f1`
mkdir -p $base
base=$base"/"$base
Pose=`head -1 $1 | awk '{print $2}'`
Sub=`head -1 $1 | awk '{print $1}'`
#Total=`awk  -v pose=$Pose -f nooftest.awk $1`

##################################################################
echo "[STEP 1]				Histogram Generation Started"
echo "                                                              "
./histogram.sh $1 $base"_GI.png" $base".sts" $base
Tgenuine=`cat G_N.dat | wc -l`
Timposter=`cat I_N.dat | wc -l`
rm -rf G_N.dat I_N.dat
#######################################################################
echo "[STEP 2]				ROC CURVE Generation Started"
echo "                                                              "
./roccurve.sh $1 $base".png" $base
#################################################################
echo "[STEP 3]	  Correct Recognition Rate (CRR) Calculation Started"
echo "                                                              "
#awk  -v total=$Total -f identification.awk $1> $base"_CRR.dat"
awk  -v PSub=$Sub -v PPose=$Pose -f identification.awk $1> $base"_CRR.dat"
correct=`awk '{if($6==1) print $0}' $base"_CRR.dat"| wc -l`
all=`awk '{print $0}' $base"_CRR.dat"| wc -l`
CRR=`calc $correct/$all*100`;
echo "Correct Recognition Rate (CRR) =   $CRR" >> $base".sts"
#################################################################
#EER=`awk -f eer.awk $base"_FAR_FRR_Data.dat"`
#echo "Equal Error Rate         (EER) = $EER" >> $base".sts"
echo "[STEP 4]		  Equal Error Rate (EER) Calculation Started"
echo "                                                              "
awk -f eer.awk $base"_FAR_FRR_Data.dat" >> $base".sts"
#################################################################
echo "[STEP 5]		  Failed Subject in Identification (CRR Failure)"
echo "                                                              "
echo "**************************************************************************************************************************************" >> $base".sts"
echo "Failed Subject in Identification (CRR Failure)" >> $base".sts"
echo "===========================" >> $base".sts"
awk '{if($6==0) print $0}' $base"_CRR.dat" >> $base".sts"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $base".sts"
failed=`awk '{if($6==0) print $0}' $base"_CRR.dat"| wc -l`
echo "Total Failed Subject =   $failed (out of $all)" >> $base".sts"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $base".sts"
echo "**************************************************************************************************************************************" >> $base".sts"
#######################################################################

#######################################################################


echo "[STEP 6]		  False Reject in Verification (ERR Genuine got Failed) HTML Report Generation"
eer=`cat $base".sts" | grep 'EER' | awk '{print $7}'`
./EER_error.sh $1 $eer 1 > $base".FR"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $base".sts"
fr=`cat $base".FR" | wc -l`
fr=`expr $fr - 2`
echo "Total No. of Falsely Reject Matching =   $fr out of total $Tgenuine genuine matching" >> $base".sts"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $base".sts"
echo "**************************************************************************************************************************************" >> $base".sts"
#######################################################################
echo "[STEP 7]		  False Accept in Verification (ERR Imposter got Accepted) HTML Report Generation"
eer=`cat $base".sts" | grep 'EER' | awk '{print $7}'`
./EER_error.sh $1 $eer 0 > $base".FA"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $base".sts"
fa=`cat $base".FA" | wc -l`
fa=`expr $fa - 2`
echo "Total No. of Falsely Accept Matching =   $fa out of total $Timposter imposter matching" >> $base".sts"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $base".sts"
echo "**************************************************************************************************************************************" >> $base".sts"
#######################################################################
#Actual EER Calculations (although the ROC curve doesn't consider at EQUAL to threshold are correct)
FAR=`calc $fa/$Timposter*100`;
FRR=`calc $fr/$Tgenuine*100`;

echo "Actual Performance Parameters" >> $base".sts"
echo "===========================" >> $base".sts"
echo "False Acceptance Rate        (FAR) = 	$FAR" >> $base".sts"
echo "False Rejection  Rate        (FAR) = 	$FRR" >> $base".sts"
#echo "Equal Error      Rate        (EER) = 	$EER with Difference = $Diff" >> $base".sts"
echo "**************************************************************************************************************************************" >> $base".sts"
#######################################################################
rm -rf *.dat
