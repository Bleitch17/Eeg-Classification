NUM_EEG_CHANNELS = 22;
NUM_EOG_CHANNELS = 3;

% From: https://www.bbci.de/competition/iv/desc_2a.pdf
EVENT_TYPE_TRIAL_START = 768;
EVENT_TYPE_REST = 276;

TRIALS_PER_SESSION = 288;

SAMPLE_RATE_HZ = 250;

% Electrodes mapping from the dataset description to the 10-10 system:
% https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)#/media/File:EEG_10-10_system_with_additional_information.svg
ELECTRODES = {
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'EOGL', 'EOGM', 'EOGR'
};