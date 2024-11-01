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
    
    true_labels = load(strcat(session_id, '.mat'));
    true_label_idx = 1;

    % First 22 columns are EEG, last 3 are EOG
    eeg_eog_data = s(:, 1:NUM_EEG_CHANNELS + NUM_EOG_CHANNELS);

    num_events = uint64(length(EVENT_TYPES));
    num_samples = uint64(h.NRec);

    data_with_events = [eeg_eog_data, zeros(num_samples, num_events)];

    for event_idx = 1:length(h.EVENT.TYP)
        event_type = h.EVENT.TYP(event_idx);

        % If the event_type is the unknown cue, then need to find the corresponding true label
        if event_type == EVENT_TYPE_UNKNOWN_CUE
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
        end

        event_pos = h.EVENT.POS(event_idx);
        event_dur = h.EVENT.DUR(event_idx);

        % Expect event_type to be the decimal number representing the event
        event_col_idx = find(EVENT_TYPES == event_type);
        
        for j = 0:event_dur
            if event_pos + j <= num_samples

                % Note - events may have 0 duration
                % Note - this assumes event_pos is 1-indexed
                data_with_events(event_pos + j, NUM_EEG_CHANNELS + NUM_EOG_CHANNELS + event_col_idx) = event_type;
            else
                warning('Event position %d with duration %d exceeds the number of samples %d', event_pos, event_dur, num_samples);
            end
        end
    end

    headers = [arrayfun(@(x) sprintf('EEG_%d', x), 1:NUM_EEG_CHANNELS, 'UniformOutput', false), ...
               arrayfun(@(x) sprintf('EOG_%d', x), 1:NUM_EOG_CHANNELS, 'UniformOutput', false), ...
               arrayfun(@(x) sprintf('Event_%d', x), EVENT_TYPES, 'UniformOutput', false)];

    % Write the data to a CSV file
    csvwrite_with_headers(strcat(session_id, '.csv'), data_with_events, headers);
end

function csvwrite_with_headers(filename, data, headers)
    fid = fopen(filename, 'w');
    fprintf(fid, '%s,', headers{1:end-1});
    fprintf(fid, '%s\n', headers{end});
    fclose(fid);

    dlmwrite(filename, data, "-append");
end