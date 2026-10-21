BEGIN {
c=0;PP=pose;
}
{
if($2 == PP)
{
c=c+1;
}
else
{
exit
}
}
END {
printf "%d\n",c;
	
}
