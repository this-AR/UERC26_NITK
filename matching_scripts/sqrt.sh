#!/bin/bash
if [ $# -ne 1 ]
then
echo 'Usage: sqrt number'
exit 1
else
#echo -e "sqrt($1)\nquit\n" | bc -q -i
a="sqrt($1)" | bc -q -i
echo $a
fi
