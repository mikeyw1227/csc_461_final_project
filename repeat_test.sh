#!/usr/bin/env bash
for i in {1..20}
do
    ./test.py > $i.txt
done
