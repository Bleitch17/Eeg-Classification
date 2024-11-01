#!/usr/bin/env bash

# Download the BCI Competition IV 2a dataset from the website:

if [ -f "BCICIV_2a_gdf.zip" ]; then
    echo "BCICIV_2a_gdf.zip already exists. Skipping download."
else
    wget https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
fi

if [ -f "true_labels.zip" ]; then
    echo "true_labels.zip already exists. Skipping download."
else
    wget https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
fi

unzip -o BCICIV_2a_gdf.zip
unzip -o true_labels.zip

# Generate one CSV file per session:
for session in A0{1..9}{E,T}; do
    if [ ! -f "$session.csv" ]; then
        octave --eval "gdf_to_csv('$session')"
    else
        echo "$session.csv already exists. Skipping conversion."
    fi
done

# Remove the GDF and MAT files:
rm *.gdf
rm *.mat