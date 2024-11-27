!!!This is an experiment branch with preprocessing get rid of the overlap. But getting super low accuracy. And we decide to still use the overlap
1) Having overlapping windows actually gives us a lot more data to work with
2) Still need to capture temporal dependencies regardless. If we have no overlap, then in a real system where our windows are 100 samples (400ms), we'd be feeding windows in every 400ms, introducing latency. With overlap, we can feed windows in much quicker.


Important notes for all models' improvements/changes:
1. With Enhanced preprocessing: Baseline correction, bandpass filter, and notch filter
2. K-fold cross validation
3. Early stopping and hyperparameter tuning
4. *For CNN -> Modified model architecture for CNN with several versions:
   - Version 2: Relu but with dropout and more regularization
   - Version 3: Leaky Relu (Theoretically better than Relu for EEG data) and added Hypertuning and early stopping
   - Version 4: ELU activation function (Theoretically better than Leaky Relu for EEG data)

5.!TODO Issue with current CNN: The Tuning and Regularization is bit too much for Version 3 to prevent overfitting. 
Possibly need to add more data augmentation. Cause the validation accuracy is less fluctuating than Version 2 but keeps higher than training accuracy. Double check if this is expected.

6. Enhanced the training progress monitor where you can visually see the progress of the training and validation accuracies.

7. *For SVM, this is newly added and I traied it once with super high accuracy 94% ish for single channels but poor performance for multi-channels. Need to do more research on and understand this though it is expected from the standpoint of SVM's machanism.
    
8. *For LSTM, Improved the model architecture and added early stopping and hyperparameter tuning.
   -Bidirectional LSTM for better temporal feature capture
   -Added dropout layer same as CNN (keep the dropout rate low as it is sensitive to overfitting)
   -made to match the CNN: Matched the input tensor shape handling, same optimization methodology
