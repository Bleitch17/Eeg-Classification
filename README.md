# Eeg-Classification
Group respository for the CPEN-355 project: EEG classification using ML techniques.

This README contains instructions for setting up the environment to work with this project.

## Conda Installation

Follow [these](https://docs.anaconda.com/anaconda/install/) steps to install the Anaconda distribution for your operating system. If installed corectly, you should be able to run this:
```
conda --version
```
and see a similar output:
```
conda 24.9.2
```

>Note - After installing Anaconda, you may notice that the base environment gets activated by default when you open a new shell terminal. To disable this, run `conda config --set auto_activate_base False`. 

To create a new conda environment, run the following command using the `ENV.yml` file in this repo:
```
conda env create -n eeg-classification --file ENV.yml
```

The activate can be activated with:
```
conda activate eeg-classification
```

More Conda commands may be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html).