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
    
    % Index to keep track of the current true label
    true_label_idx = 1;

    % First 22 columns are EEG, last 3 are EOG
    eeg_eog_data = num2cell(s(:, 1:NUM_EEG_CHANNELS + NUM_EOG_CHANNELS));

    % Add two extra columns: one for the event types and one for the event durations
    % Note - it is possible for multiple events to map to the same sample, hence using cell arrays
    empty_col = cell(size(eeg_eog_data, 1), 1);
    for empty_col_idx = 1:size(empty_col, 1)
        empty_col{empty_col_idx} = [];
    end
    
    data_with_events = [eeg_eog_data, empty_col, empty_col];

    for event_idx = 1:length(h.EVENT.TYP)
        event_type = h.EVENT.TYP(event_idx);

        % If the event_type is the unknown cue, then need to find the corresponding true label
        if event_type == EVENT_TYPE_UNKNOWN_CUE
            assert(true_label_idx <= TRIALS_PER_SESSION, 'Exceeded the number of expected trials in the session');
            
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
        
        % Store the event type and duration in the last two columns
        data_with_events{event_pos, EVENT_TYPES_COL} = [data_with_events{event_pos, EVENT_TYPES_COL}, event_type];
        data_with_events{event_pos, EVENT_DURATIONS_COL} = [data_with_events{event_pos, EVENT_DURATIONS_COL}, event_dur];
    end

    headers = [arrayfun(@(x) sprintf('EEG_%d', x), 1:NUM_EEG_CHANNELS, 'UniformOutput', false), ...
               arrayfun(@(x) sprintf('EOG_%d', x), 1:NUM_EOG_CHANNELS, 'UniformOutput', false), ...
               'Event Types', 'Event Durations'];

    % Write the data to a CSV file
    csvwrite_with_headers(strcat(session_id, '.csv'), data_with_events, headers);
end

function csvwrite_with_headers(filename, data, headers)
    constants;
    
    fid = fopen(filename, 'w');
    fprintf(fid, '%s,', headers{1:end-1});
    fprintf(fid, '%s\n', headers{end});
    
    for row_idx = 1:size(data, 1)
        % Write the EEG and EOG data
        for col_idx = 1:NUM_EEG_CHANNELS + NUM_EOG_CHANNELS - 1
            fprintf(fid, '%f,', data{row_idx, col_idx}(1));
        end
        fprintf(fid, '%f', data{row_idx, NUM_EEG_CHANNELS + NUM_EOG_CHANNELS}(1));
        
        num_events = length(data{row_idx, EVENT_TYPES_COL});

        if num_events == 0
            for ignore = 1:2
                fprintf(fid, ',');
            end
            fprintf(fid, '\n');

        elseif num_events == 1
            fprintf(fid, ',%d,%d\n', data{row_idx, EVENT_TYPES_COL}(1), data{row_idx, EVENT_DURATIONS_COL}(1));

        else
            fprintf(fid, ',');

            for event_idx = 1:num_events - 1
                fprintf(fid, '%d ', data{row_idx, EVENT_TYPES_COL}(event_idx));
            end

            fprintf(fid, '%d,', data{row_idx, EVENT_TYPES_COL}(end));

            for event_idx = 1:num_events - 1
                fprintf(fid, '%d ', data{row_idx, EVENT_DURATIONS_COL}(event_idx));
            end

            fprintf(fid, '%d\n', data{row_idx, EVENT_DURATIONS_COL}(end));
        end
    end

    fclose(fid);
end