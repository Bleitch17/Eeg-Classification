function gdf_to_csv(session_id)
    % Input file name should follow the pattern "A0[1-9][T-E]" as seen in the dataset
    if isempty(regexp(session_id, '^A0[1-9][E-T]$', 'once'))
        error('Input file name does not match the required pattern "A0[1-9][T-E]"');
    end
    
    % Need sload from the biosig package
    pkg load biosig;
    
    % Loads constants from separate file
    constants;

    [s, h] = sload(strcat(session_id, '.gdf'));
    
    % Expecting true_labels to hold [288 x 1] class labels
    true_labels = load(strcat(session_id, '.mat'));

    % EEG and EOG data matrix with 27 columns:
    % 22 EEG columns
    % 3 EOG columns
    % 1 Label column
    % 1 Recording Index column
    eeg_eog_data = zeros(size(s, 1), NUM_EEG_CHANNELS + NUM_EOG_CHANNELS + 2);

    % New event matrix will have columns: [label, position, duration]
    event_matrix = create_event_matrix(h.EVENT, true_labels.classlabel, h.ArtifactSelection);

    % Recordings will be 0-indexed for the CSV file
    recording_index = 0;

    sample_index = 1;

    for event_index = 1:size(event_matrix, 1)
        event_label = event_matrix(event_index, 1);
        event_position = event_matrix(event_index, 2);
        event_duration = event_matrix(event_index, 3);

        for event_sample_index = event_position:event_position + event_duration
            % Only update the recording index if the current sample contains NaN values,
            % and the previous sample does not. If there is no previous sample, i.e.: the first sample
            % is NaN, then no need to update the recording index.
            if any(isnan(s(event_sample_index, :)))
                if event_sample_index > 1 && ~any(isnan(s(event_sample_index - 1, :)))
                    recording_index = recording_index + 1;
                end
                continue;
            end

            % Get the current row from the original data, and place it in the new data matrix
            eeg_eog_data(sample_index, 1:NUM_EEG_CHANNELS + NUM_EOG_CHANNELS) = s(event_sample_index, 1:NUM_EEG_CHANNELS + NUM_EOG_CHANNELS);
            eeg_eog_data(sample_index, NUM_EEG_CHANNELS + NUM_EOG_CHANNELS + 1) = event_label;
            eeg_eog_data(sample_index, NUM_EEG_CHANNELS + NUM_EOG_CHANNELS + 2) = recording_index;

            sample_index = sample_index + 1;
        end

        % If the last sample of the event was NaN, the recording index would already have been updated
        if ~any(isnan(s(event_position + event_duration, :)))
            recording_index = recording_index + 1;
        end
    end

    % Trim the data to the correct size
    eeg_eog_data = eeg_eog_data(1:sample_index - 1, :);

    headers = [ELECTRODES, 'Label', 'Recording'];

    % Write the data to a CSV file
    csvwrite_with_headers(strcat(session_id, '.csv'), headers, eeg_eog_data);
end


function event_matrix = create_event_matrix(event_struct, true_labels, artifact_select)
    constants;
    
    % The event matrix will not have more rows than the number of events in the event struct
    event_matrix = zeros(length(event_struct.TYP), 3);    
    new_event_index = 1;
    trial_index = 1;

    for event_index = 1:length(event_struct.TYP)
        event_type = event_struct.TYP(event_index);
        event_pos = event_struct.POS(event_index);
        event_dur = event_struct.DUR(event_index);

        if event_type == EVENT_TYPE_REST
            event_type = 0;
        
        elseif event_type == EVENT_TYPE_TRIAL_START
            % If the current trial has an artifact, skip it
            if artifact_select(trial_index) == 1
                trial_index = trial_index + 1;
                continue;
            end
            
            event_type = true_labels(trial_index);
            
            % The samples of interest for the trial are from [t = 3s, t = 6s)
            % Therefore, includes sample positions [751, 1500]
            event_pos = event_pos + 3 * SAMPLE_RATE_HZ;

            % NOTE - event durations don't include the sample at the start of the event, i.e.: they are
            % the number of samples to read in addition to the starting sample.
            % Therefore, to get 750 samples per trial, need to read 749 additional samples.
            event_dur = 3 * SAMPLE_RATE_HZ - 1;

            trial_index = trial_index + 1;

        else
            % Skip all other events
            continue;
        end

        event_matrix(new_event_index, 1) = event_type;
        event_matrix(new_event_index, 2) = event_pos;
        event_matrix(new_event_index, 3) = event_dur;

        new_event_index = new_event_index + 1;
    end

    % Trim the event matrix to the correct size
    event_matrix = event_matrix(1:new_event_index - 1, :);
end


function csvwrite_with_headers(filename, headers, eeg_eog_data)
    constants;
    
    % Open the file for writing
    fid = fopen(filename, 'w');
    
    % Write the headers
    fprintf(fid, '%s,', headers{1:end-1});
    fprintf(fid, '%s\n', headers{end});
    
    % Write the data
    for row = 1:size(eeg_eog_data, 1)
        fprintf(fid, '%f,', eeg_eog_data(row, 1:end-1));
        fprintf(fid, '%f\n', eeg_eog_data(row, end));
    end
    
    % Close the file
    fclose(fid);
end