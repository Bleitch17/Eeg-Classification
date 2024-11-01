# Eeg-Classification
Group respository for the CPEN-355 project: EEG classification using ML techniques.

This README contains instructions for setting up the environment to work with this project.

## Conda Installation

Follow [these](https://docs.anaconda.com/anaconda/install/) steps to install the Anaconda distribution for your operating system. If installed corectly, you should be able to run this:
```
conda --version
```
And see a similar output:
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

More Conda commands may be found in the cheat sheet [here](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html).

## GNU Octave Installation

Many EEG datasets come in `.mat` files or other filetypes that require specific data processing software to open. While MATLAB can be used to process these files, GNU Octave provides a free, open-source solution to handle these files as well.

Install GNU Octave for your operating system by downloading the correct installer for your operating system [here](https://octave.org/download).

To verify that GNU Octave has successfully been installed, run the following:
```
octave --version
```

You should see an output similar to:
```
GNU Octave, version 8.4.0
Copyright (C) 1993-2023 The Octave Project Developers.
This is free software; see the source code for copying conditions.
There is ABSOLUTELY NO WARRANTY; not even for MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.

Octave was configured for "x86_64-pc-linux-gnu".

Additional information about Octave is available at https://www.octave.org.

Please contribute if you find this software useful.
For more information, visit https://www.octave.org/get-involved.html

Read https://www.octave.org/bugs.html to learn how to submit bug reports.
```

>Note - Windows users will likely need to add the GNU Octave `bin` directory to their path, for the `octave` command to be recognized from the terminal. The default installation location may vary: if Octave was installed system-wide, then `C:\Octave\Octave-<version>\mingw64\bin` would be a likely place to check.

To access EEG datasets saved in `.gdf` format, the `biosig` package can be installed in Octave, which provides the `sload(...)` function.
To install, run (in Octave):
```
pkg install -forge biosig
```

After the package has installed, you can access its functions by loading it:
```
pkg load biosig
```
