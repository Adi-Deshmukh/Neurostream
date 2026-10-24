[![DOI](https://img.shields.io/badge/DOI-10.82901%2Fnemar.nm000139-blue)](https://doi.org/10.82901/nemar.nm000139)

# BNCI 2014-001 Motor Imagery dataset

BNCI 2014-001 Motor Imagery dataset.

## Dataset Overview

- **Code**: BNCI2014-001
- **Paradigm**: imagery
- **DOI**: 10.3389/fnins.2012.00055
- **Subjects**: 9
- **Sessions per subject**: 2
- **Events**: left_hand=1, right_hand=2, feet=3, tongue=4
- **Trial interval**: [2, 6] s
- **Runs per session**: 6
- **File format**: GDF
- **Data preprocessed**: True

## Acquisition

- **Sampling rate**: 250.0 Hz
- **Number of channels**: 25
- **Channel types**: eeg=22, eog=3
- **Channel names**: C1, C2, C3, C4, C5, C6, CP1, CP2, CP3, CP4, CPz, Cz, EOG1, EOG2, EOG3, FC1, FC2, FC3, FC4, FCz, Fz, P1, P2, POz, Pz
- **Montage**: custom
- **Hardware**: BrainAmp MR plus
- **Software**: BCI2000
- **Reference**: left mastoid
- **Ground**: unknown
- **Sensor type**: Ag/AgCl
- **Line frequency**: 50.0 Hz
- **Online filters**: bandpass 0.05-200 Hz
- **Cap manufacturer**: EASYCAP GmbH

## Participants

- **Number of subjects**: 9
- **Health status**: healthy
- **Species**: human

## Experimental Protocol

- **Paradigm**: imagery
- **Number of classes**: 4
- **Class labels**: left_hand, right_hand, feet, tongue
- **Trial duration**: 4.0 s
- **Study design**: Two-class motor imagery (selected from left hand, right hand, and foot) with asynchronous/continuous control periods
- **Feedback type**: none
- **Stimulus type**: arrow_cue
- **Stimulus modalities**: visual, auditory
- **Primary modality**: multisensory
- **Synchronicity**: asynchronous
- **Mode**: offline
- **Instructions**: Subjects instructed to perform motor imagery during cued periods

## HED Event Annotations

Schema: HED 8.4.0 | Browse: https://www.hedtags.org/hed-schema-browser

```
  left_hand
    ├─ Sensory-event
    │  ├─ Experimental-stimulus
    │  ├─ Visual-presentation
    │  └─ Leftward, Arrow
    └─ Agent-action
       └─ Imagine
          ├─ Move
          └─ Left, Hand

  right_hand
    ├─ Sensory-event
    │  ├─ Experimental-stimulus
    │  ├─ Visual-presentation
    │  └─ Rightward, Arrow
    └─ Agent-action
       └─ Imagine
          ├─ Move
          └─ Right, Hand

  feet
    ├─ Sensory-event
    │  ├─ Experimental-stimulus
    │  ├─ Visual-presentation
    │  └─ Downward, Arrow
    └─ Agent-action
       └─ Imagine, Move, Foot

  tongue
    ├─ Sensory-event
    │  ├─ Experimental-stimulus
    │  ├─ Visual-presentation
    │  └─ Upward, Arrow
    └─ Agent-action
       └─ Imagine, Move, Tongue

```
## Paradigm-Specific Parameters

- **Detected paradigm**: motor_imagery
- **Imagery tasks**: left_hand, right_hand, foot
- **Cue duration**: 4.0 s
- **Imagery duration**: 4.0 s

## Data Structure

- **Trials**: {'training': 200, 'test': 240}
- **Blocks per session**: 6
- **Trials context**: per subject (2 training runs + 4 test runs)

## Preprocessing

- **Data state**: minimally preprocessed (bandpass and notch filtered)
- **Preprocessing applied**: True
- **Steps**: bandpass filtering
- **Highpass filter**: 0.05 Hz
- **Lowpass filter**: 200 Hz
- **Bandpass filter**: {'low_cutoff_hz': 0.05, 'high_cutoff_hz': 200.0}
- **Filter type**: analog
- **Re-reference**: none
- **Downsampled to**: 100.0 Hz
- **Notes**: Data provided in two versions: original at 1000 Hz and downsampled to 100 Hz (with Chebyshev Type II filter order 10, stop band ripple 50 dB, stop band edge 49 Hz)

## Signal Processing

- **Classifiers**: LDA, SVM, Neural Network, Naive Bayes, RBF Neural Network
- **Feature extraction**: CSP, FBCSP, Bandpower, ERD, ERS
- **Frequency bands**: mu=[8, 12] Hz; beta=[16, 24] Hz

## Cross-Validation

- **Method**: train-test split
- **Evaluation type**: within_session

## Performance (Original Study)

- **Mse**: 0.382

## BCI Application

- **Applications**: cursor_control, communication
- **Environment**: laboratory
- **Online feedback**: False

## Tags

- **Pathology**: Healthy
- **Modality**: Motor
- **Type**: Motor

## Documentation

- **Description**: Review of the BCI competition IV - Data set 1: Asynchronous Motor Imagery
- **DOI**: 10.3389/fnins.2012.00055
- **License**: CC-BY-ND-4.0
- **Investigators**: Michael Tangermann, Klaus-Robert Müller, Ad Aertsen, Niels Birbaumer, Christoph Braun, Clemens Brunner, Robert Leeb, Carsten Mehring, Kai J. Miller, Gernot R. Müller-Putz, Guido Nolte, Gert Pfurtscheller, Hubert Preissl, Gerwin Schalk, Alois Schlögl, Carmen Vidaurre, Stephan Waldert, Benjamin Blankertz
- **Senior author**: Michael Tangermann
- **Contact**: michael.tangermann@tu-berlin.de
- **Institution**: Berlin Institute of Technology
- **Department**: Machine Learning Laboratory
- **Address**: FR 6-9, Franklinstr. 28/29, 10587 Berlin, Germany
- **Country**: AT
- **Repository**: BNCI Horizon
- **Data URL**: http://www.bbci.de/competition/iv/
- **Publication year**: 2012
- **Keywords**: brain-computer interface, BCI, competition

## References

Tangermann, M., Muller, K.R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G. and Nolte, G., 2012. Review of the BCI competition IV. Frontiers in neuroscience, 6, p.55.

Notes

.. note::

``BNCI2014_001`` was previously named ``BNCI2014001``. ``BNCI2014001`` will be removed in version 1.1.

.. versionadded:: 0.4.0

This is one of the most widely used motor imagery datasets in BCI research, commonly referred to as "BCI Competition IV Dataset 2a". It serves as a standard benchmark for 4-class motor imagery classification algorithms.

The dataset is particularly useful for:

- Multi-class motor imagery classification (4 classes) - Transfer learning studies (9 subjects, 2 sessions each) - Cross-session variability analysis

See Also

BNCI2014_004 : BCI Competition 2008 2-class motor imagery (Dataset B) BNCI2003_004 : BCI Competition III 2-class motor imagery

Examples

>>> from moabb.datasets import BNCI2014_001 >>> dataset = BNCI2014_001() >>> dataset.subject_list [1, 2, 3, 4, 5, 6, 7, 8, 9]
Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Hochenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896

Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 6, 103. https://doi.org/10.1038/s41597-019-0104-8

---
Generated by MOABB 1.4.3 (Mother of All BCI Benchmarks)
https://github.com/NeuroTechX/moabb
