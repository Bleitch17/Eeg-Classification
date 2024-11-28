# EEG Classification - Experments, Models Exploration, and Advancements in Deep Learning CNN for EEG

## Table of Contents
- [1. Data Processing Rationale and Workflow](#1-data-processing-rationale-and-workflow)
  - [1.1 Dataset Overview and Challenges](#11-dataset-overview)
  - [1.2 Experiments and Data Processing Selection](#12-experiments-and-data-processing-selection)

- [2. Models & Evaluations (From Worst to Best)](#2-models--evaluations)
  - [2.0 Simple Classifiers](#20-simple-classifiers)
    - [2.0.1 Naive Bayes](#201-naive-bayes)
    - [2.0.2 Support Vector Machine (SVM)](#202-support-vector-machine-svm)
    - [2.0.3 Random Forest](#203-random-forest)
  - [2.1 *Key Part*:Model Architectures for LSTM and normal + deeper learning CNNs](#21-model-architectures)
    - [2.1.0 LSTM](#210-lstm)
    - [2.1.1 Basic CNN with ReLU](#211-basic-cnn-with-relu)
    - [2.1.2 Basic CNN with ELU](#212-basic-cnn-with-elu)
    - [2.1.3 Deep CNN](#213-deep-cnn)
  - [2.2 Training Process](#22-training-process)
  - [2.3 Performance Comparison](#23-performance-comparison)

- [3. How to Read and Use our Repository](#3-how-to-read-and-use-our-repository)
  - [3.1 Repository Structure](#31-repository-structure)
  - [3.2 Installation Guide](#32-installation-guide)
  - [3.3 Usage Instructions](#33-usage-instructions)
  - [3.4 Results Visualization](#34-results-visualization)

## 1. Data Processing Rationale and Workflow
Due to the size and complexity of the problem, we will focus on the (BCI Competition IV Dataset 2a)(https://archive.ics.uci.edu/dataset/533/eeg+database) Subject 1's data for this project.
As a result of careful experiments, we choose to use 5-fold cross-validation with no normalization in preprocessing (we did normalization in complex models), bandpass 8 - 50Hz, window size 100, and window overlap 90.

### 1.1 Dataset Overview and Challenges

#### Overview
The BCI Competition IV Dataset 2a contains EEG recordings from 9 subjects performing motor imagery tasks. Our analysis focuses on Subject 1's data:
- 22 EEG channels
- 5 motor imagery classes: Rest(0), Left Hand(1), Right Hand(2), Feet(3), Tongue(4)
- ~480,000 EEG samples after preprocessing
- Data windowing: 100 samples/window with 90-sample overlap

[![Dataset Distribution](./Dataset%20EDA/dataset%20distribution%20hotmap.png)![Class Balance](./Dataset%20EDA/dataset_balance.png)]

#### Key Challenges
1. **Signal Quality**:
   - High noise-to-signal ratio in EEG recordings
   - Susceptibility to motion artifacts and electrical interference

2. **Temporal Dependencies**:
   - Complex temporal patterns in brain signals
   - Window overlap creates potential data leakage

3. **Class Slight Imbalance**:
   - Rest state (Class 0) underrepresented
   - Potential bias in model training

4. **Inter-trial Variability**:
   - Mental states vary between recordings
   - Inconsistent signal patterns for same motor imagery tasks

### 1.2 Experiments and Data Processing Selections
#### Rationale for Choices
These choices were made based on experimental results and represent an optimal balance between model performance and computational efficiency.
(*Note: if interested please see Experimental Results folder)

1. **5-fold Cross-validation**:
   - Provides robust model evaluation
   - Balances between computational cost and validation reliability
   - Helps detect and prevent overfitting

2. **Window Size and Overlap**:
   - Window Size (100): Captures sufficient temporal patterns
   - Overlap (90%): 
     - Significantly improves classification accuracy (~10% increase)
     - Enables faster real-time predictions
     - Provides more training samples while preserving temporal continuity
   - Models showed poor performance without overlap due to missed temporal transitions

3. **Bandpass Filter (8-50Hz)**:
   - Removes low-frequency artifacts
   - Retains relevant motor imagery frequencies
   - Eliminates high-frequency noise

4. **No Initial Normalization**:
   - Preserves original signal characteristics
   - Normalization performed within model architectures
   - Allows models to learn from raw signal patterns

## 2. Models & Evaluations (From Worst to Best)
[Content for this section...]

### 2.0 Simple Classifiers
[Content for this section...]

#### 2.0.1 Naive Bayes
[Content for this section...]

#### 2.0.2 Support Vector Machine (SVM)
[Content for this section...]

#### 2.0.3 Random Forest
[Content for this section...]

### 2.1  *Key Part*: Model Architectures for LSTM and normal + deeper learning CNNs
[Content for this section...]

#### 2.1.0 LSTM
[Content for this section...]

#### 2.1.1 Basic CNN with ReLU
[Content for this section...]

#### 2.1.2 Basic CNN with ELU
[Content for this section...]

#### 2.1.3 Deep CNN
[Content for this section...]

### 2.2 Training Process
[Content for this section...]

### 2.3 Performance Comparison
[Content for this section...]

## 3. How to Read and Use our Repository
[Content for this section...]

### 3.1 Repository Structure
[Content for this section...]

### 3.2 Installation Guide
[Content for this section...]

### 3.3 Usage Instructions
[Content for this section...]

### 3.4 Results Visualization
[Content for this section...]
