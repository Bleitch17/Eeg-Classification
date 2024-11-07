#!/usr/bin/env bash

NUM_SUBJECTS=1
if [ $# -gt 0 ]; then
    NUM_SUBJECTS=$1
fi

if [ $NUM_SUBJECTS -lt 1 ] || [ $NUM_SUBJECTS -gt 9 ]; then
    echo "Number of subjects must be between 1 and 9."
    exit 1
fi

# Download the BCI Competition IV 2a dataset from the website:
if [ -f "BCICIV_2a_gdf.zip" ]; then
    echo "BCICIV_2a_gdf.zip already exists. Skipping download."
else
    curl -kLSs https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip -o BCICIV_2a_gdf.zip
fi

if [ -f "true_labels.zip" ]; then
    echo "true_labels.zip already exists. Skipping download."
else
    curl -kLSs https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip -o true_labels.zip
fi

unzip -o BCICIV_2a_gdf.zip
unzip -o true_labels.zip

# Generate one CSV file per session for the specified number of subjects:
for subject in $(seq -w 1 $NUM_SUBJECTS); do
    for session in A0${subject}{E,T}; do
        if [ ! -f "$session.csv" ]; then
            octave --eval "gdf_to_csv('$session')"
        else
            echo "$session.csv already exists. Skipping conversion."
        fi
    done
done

# Remove the GDF and MAT files:
rm *.gdf
rm *.mat