#!/usr/bin/env bash

res="res"

if ! [[ -d "./$res" ]]; then
    mkdir ${res}
    cd ./${res}
    kaggle competitions download -c LANL-Earthquake-Prediction
    unzip train.csv.zip
    unzip test.zip
    cd ..
fi
echo "Data donwloaded"

cd ./${res}

if ! [[ -f "./train_without_header.csv" ]]; then
    tail -n +2 train.csv >> train_without_header.csv
fi

echo "CSV header removed"

train_splitted="train_splitted"

if ! [[ -d "./$train_splitted" ]]; then
    mkdir ${train_splitted}
    cd ./${train_splitted}
    split -l 2000000 --numeric-suffixes ../train_without_header.csv train_splitted_
    cd ..
fi

echo "Data splitted"

cd ..

echo "Will produce fft and pickle files..."

python pickle_data.py

echo "Finished"
