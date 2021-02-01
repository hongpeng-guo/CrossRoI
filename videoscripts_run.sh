#!/bin/bash

if [ $1 == 'crop' ]
then
    for f in scripts/crop/*.sh; do
        bash "$f" -H
    done
elif [ $1 == 'merge' ]
then
    for f in scripts/merge/*.sh; do
        bash "$f" -H
    done
else
    echo "please input crop or merge"
fi