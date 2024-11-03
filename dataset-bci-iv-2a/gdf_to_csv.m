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
    
    % Expecting artifacts to hold [288 x 1] booleans:
    % 1 if the trial contains an artifact, 0 otherwise.
    artifacts = h.ArtifactSelection;

    % Indexes to track current trial true labal, and current trial artifact status
    true_label_idx = 1;
    artifact_idx = 1;

    % First 22 columns are EEG, last 3 are EOG
    eeg_eog_data = s(:, 1:NUM_EEG_CHANNELS + NUM_EOG_CHANNELS);

    % Next column will be artifact status
    artifact_status = zeros(size(eeg_eog_data, 1), 1);

    % Next two columns will be event types and event durations
    % Note - it is possible for multiple events to map to the same sample, hence using cell arrays
    event_types = cell(size(eeg_eog_data, 1), 1);
    for event_types_idx = 1:size(event_types, 1)
        event_types{event_types_idx} = [];
    end

    event_durations = cell(size(eeg_eog_data, 1), 1);
    for event_durations_idx = 1:size(event_durations, 1)
        event_durations{event_durations_idx} = [];
    end
    
    for event_idx = 1:length(h.EVENT.TYP)
        event_type = h.EVENT.TYP(event_idx);
        event_pos = h.EVENT.POS(event_idx);
        event_dur = h.EVENT.DUR(event_idx);

        % If the event_type is the unknown cue, then need to find the corresponding true label
        if event_type == EVENT_TYPE_UNKNOWN_CUE
            assert(true_label_idx <= TRIALS_PER_SESSION, 'True label select exceeded the number of expected trials in the session');
            
            class_label = true_labels.classlabel(true_label_idx);
            
            % Class labels to events: https://www.bbci.de/competition/iv/desc_2a.pdf
            if class_label == 1
                event_type = EVENT_TYPE_CUE_LEFT;
            elseif class_label == 2
                event_type = EVENT_TYPE_CUE_RIGHT;
            elseif class_label == 3
                event_type = EVENT_TYPE_CUE_FEET;
            elseif class_label == 4
                event_type = EVENT_TYPE_CUE_TONGUE;
            end
            
            true_label_idx = true_label_idx + 1;
        
        elseif event_type == EVENT_TYPE_TRIAL_START
            assert(artifact_idx <= TRIALS_PER_SESSION, 'Artifact status select exceeded the number of expected trials in the session');
            
            % Since artifacts column starts with all 0's anyways, just set the value to the artifact status
            artifact_status(event_pos) = artifacts(artifact_idx);
            artifact_idx = artifact_idx + 1;
        end

        % Append the event type to the event types array at the event position
        event_types{event_pos} = [event_types{event_pos}, event_type];

        % Append the event duration to the event durations array at the event position
        event_durations{event_pos} = [event_durations{event_pos}, event_dur];
    end

    headers = [ELECTRODES, 'Artifact Status', 'Event Types', 'Event Durations'];

    % Write the data to a CSV file
    csvwrite_with_headers(strcat(session_id, '.csv'), headers, eeg_eog_data, artifact_status, event_types, event_durations);
end


function csvwrite_with_headers(filename, headers, eeg_eog_data, artifact_status, event_types, event_durations)
    constants;
    
    fid = fopen(filename, 'w');
    fprintf(fid, '%s,', headers{1:end-1});
    fprintf(fid, '%s\n', headers{end});
    
    % All input matrices should have the same number of rows
    assert(size(eeg_eog_data, 1) == size(artifact_status, 1), 'Number of rows in EEG/EOG data and artifact status do not match');
    assert(size(eeg_eog_data, 1) == size(event_types, 1), 'Number of rows in EEG/EOG data and event types do not match');
    assert(size(eeg_eog_data, 1) == size(event_durations, 1), 'Number of rows in EEG/EOG data and event durations do not match');

    for row_idx = 1:size(eeg_eog_data, 1)
        % Write the EEG and EOG data
        for col_idx = 1:NUM_EEG_CHANNELS + NUM_EOG_CHANNELS
            fprintf(fid, '%f,', eeg_eog_data(row_idx, col_idx));
        end

        fprintf(fid, '%d', artifact_status(row_idx));     

        num_events = length(event_types{row_idx});

        if num_events == 0
            for ignore = 1:2
                fprintf(fid, ',');
            end
            fprintf(fid, '\n');

        elseif num_events == 1
            fprintf(fid, ',%d,%d\n', event_types{row_idx}(1), event_durations{row_idx}(1));

        else
            fprintf(fid, ',');

            for event_idx = 1:num_events - 1
                fprintf(fid, '%d ', event_types{row_idx}(event_idx));
            end

            fprintf(fid, '%d,', event_types{row_idx}(end));

            for event_idx = 1:num_events - 1
                fprintf(fid, '%d ', event_durations{row_idx}(event_idx));
            end

            fprintf(fid, '%d\n', event_durations{row_idx}(end));
        end
    end

    fclose(fid);
end