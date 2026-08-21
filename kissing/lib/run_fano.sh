#!/bin/bash
# Long-running enumeration of global codes V for the dim-14 Fano-product design.
cd /home/user/unsolved
for s in $(seq 300 400); do
  python3 kissing/dim14/fano50c.py $s 600 >> kissing/logs/d14_fano.log 2>&1
done
