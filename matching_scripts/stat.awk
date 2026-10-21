#
#  AWK SCRIPT stat does statistics... prints out arithmetic mean,
#  standard deviation, and standard error of column 2
#

BEGIN { n = 0; s = 0; ss = 0;}
      { n++; s += $2; ss += $2 *$2; }
END {
  print n " data points"
  m = (s+0.0)/n; print m " average"
  sd = sqrt( (ss - n * m * m) / ( n - 1.0))
  print sd " standard deviation"
  se = sd/sqrt(n)
  print se " standard error"
}
