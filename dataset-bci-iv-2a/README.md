# Downloading the BCI Competition IV 2A Dataset

The BCI Competition IV 2A Dataset is saved in `.gdf` format. To create `.csv` files for the data, ensure you have Octave and the `biosig` package installed on your system, then run the following script in `dataset-bci-iv-2a`:
```
./init_data.sh
```

This will download the zipped data files, unzip them, and convert each session into a csv file. The dataset format and event types are described in this [pdf](https://www.bbci.de/competition/iv/desc_2a.pdf).

>Note - We cannot consider all the rows which correspond to a desired cue onset event as belonging to a class of Motor Imagery, because the cue onset event does not capture the window of time the subject has been asked to perform the Motor Imagery task for. From the dataset description, the cue appears at t = 2s within the trial, and disappears at t = 6s within the trial. The run structure indicates we can safely assume that the Motor Imagery task begins at t = 3 within the trial, and ends at t = 6 within the trial. Given a sample rate of 250 Hz, that means we have 750 samples per run. Some extra processing on the `csv` files will be required to capture this; the `csv` files are intended to store all the information from the data in a more friendly to parse format.
