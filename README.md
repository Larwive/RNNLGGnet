# RNNLGGnet

An attempt to classify patients with RBD (REM sleep behaviour disorder) and predict whether an RBD patient will develop
the Parkinson disease (PD) using an RNN-augmented LGGNet with PSG data.

The code architecture and the 'LGGnet' model is taken from `yi-ding-cs`'s
repository (https://github.com/yi-ding-cs/LGG).
You probably want to check their repository for better understanding.

I added an RNN part (a Gru layer) and made the training in 3 phases.

- 1st phase: Train the non-RNN part (LGGNet) only
- 2nd phase: Train the RNN part only
- 3rd phase: Train the whole model

A resnet50-inspired model is also used to compare with `RNNLGGnet`. This model is adapted to take the same inputs as
`RNNLGGnet`.

## Data preprocessing

The first considered dataset is from a sleep centre, containing whole-night EEG, EOG (electrooculogram),
ECG (electrocardiogram) and EMG (electromyogram) recordings of patients suffering from REM sleep
behaviour disorder (RBD). Some of these patients were later diagnosed with PD, and the dataset does not
include healthy subjects. This dataset is not publicly available. This dataset alone can only train neural networks to
predict whether an RBD
sufferer will develop PD. This is why the DREAMS database (https://zenodo.org/records/2650142#.ZFJ0pnbMJD8), especially
the DREAMS Subjects
Database, which comprises whole-night EEG and ECG data from healthy subjects is added to complete
the dataset. This enables the detection of RBD in comparison to healthy patients.

A drawback from using multiple datasets simultaneously is data incompatibility. **Therefore, data is
preprocessed through a band-pass filter (0.5-50 Hz), resampled to 128 Hz, and finally normalised.** An optional ICA is
present in the code. The data is cut into epochs of 4 seconds (512 points) by default.

The dataset from the sleep centre includes the channels F3-A2, F4-A1, C3-A2, C4-A1, O1-A2, O2-A1, LOC-A2, EEG ROC-A1,
EMG Chin, ECG EKG, EMG Left_Leg, EMG Right_Leg, Snoring Snore, Airflow, Resp Thorax, Resp Abdomen, Manual. Those are the
channels used for PD detection.

However, because the recordings (sleep center and DREAMS) do not include the same channels, RBD detection is limited to
**only three channels (O1, O2, and ECG) are used for RBD detection**.

## 1st phase (LGGnet)

I added a sigmoid layer at the end of the `forward` method of the model and the `fc` layer is augmented by an
intermediate `Linear` layer.
It is trained exactly as in the original repository otherwise, except for the early stopping condition. In order to
reduce overfitting risks, the early stopping condition now use the validation set instead of the unused data from the
inner 3-fold (which is from the same subject whom training data belong to).

The data is shuffled between subjects while training.

## 2nd phase (RNN part only)

A bigger `RNNLGGNet` model is being trained.
The output of the `GCN` is used for the `GRU` layer and the `fc` layer from phase 1 (named `fcLGG`).
The output of the `GRU` layer goes into another `fc` layer.
Then, the outputs of the `fcLGG` and `fc` go into a `Linear` (named `out_weights`) layer before the `sigmoid`.

This phase only trains the `GRU`, `fc`, `fcLGG` and the final `Linear` layers.

Since the `GRU` layer's hidden state is intended to be specific to each subject, data isn't shuffled while training and
linearly given to the model.
Then the hidden state is reset at each new subject.

## 3rd phase (entire model)

All the parameters are being trained simultaneously. Everything else is like the 2nd phase.

## Training of resnet-inspired model

The training of the `resnet` model is the same as the modified first stage of `RNNLGGnet`.

## Reasoning and considerations

`RNNLGGnet` is an RNN-augmented `LGGnet`. Recurrent neural networks (RNN) are a type
of neural network designed to handle time-series data, which is a data variable in length. It can also retain
information to influence future predictions. The most well-known are long short-term memory networks (LSTM) and gated
recurrent units (GRU). The choice of the RNN was straightforward as LSTM and GRU have similar performances while GRU
requires fewer parameters, thus making it easier to train.

The multiphase pipeline is designed to give sense to the different parts of the model. The `LGGnet` is meant to focus on
learning to recognise patterns while the RNN part tries to keep the relevant information depending on the past data.Due
to the presence of RNN, phase 2 and 3 have the input training data presented linearly subject per subject, resetting the
RNN part between each subject. This choice is motivated by the will of letting the model trying to predict from an
entire time-series input as a physician would do.

`ResNet`, the ResNet-50-inspired model is developed to handle 1D data, especially the time-series data provided. While
`RNNLGGnet` aims at predicting over the whole subject’s segment, `ResNet` will attempt to recognise the pattern
associated with either RBD or PD. It is indeed well known that ResNet-50 networks perform very well in image
classification.

# Results

Short version: Don't train phases 2 and 3. Phase 1 is largely better (which actually consists of a slightly less
overfitting `LGGnet`).

Link to results: https://docs.google.com/document/d/1E_bEYQ98wCeGFsEgza_1uB-pqniT3MLgYNjN5CrfDJs

**Note that the `enhanced` mention means that the phase 1 is the modified one.**

- RBD

The performance is assessed using a subject-wise cross-validation. It means that for each fold, a certain
percentage of subjects are excluded from the training set and are used as the validation set. In the
following results, a 20% excluded subjects rate is applied on the 39 subjects for RBD (both DREAMS
and sleep centre datasets) and the 19 subjects for the PD (sleep centre dataset only).

`RNNLGGnet` easily reaches 99% of accuracy when predicting whether a subject is suffering from RBD after
phase 1. Phase 2 actually reduces accuracy to below 50% and tends to make the model favour one output
over the other (0 or 1 for the presence of RBD). Phase 3 can have different behaviour, it was able to fix
the bias of phase 2 as well as going deeper in the bias and almost only giving one type of output.

As a result, it is more efficient to only keep the slightly modified LGGnet training step. The high
accuracy might also come from the distinct datasets since data from DREAMS is labelled as 0 (healthy)
while data from sleep centre is labelled as 1 (have RBD) even if everything have been normalised. A
whole dataset from the same conditions will be crucial to have trustable results.

The adapted `ResNet-50` has similar results when detecting RBD as `LGGnet`, which is to easily reach 99%
of accuracy. The resemblance doesn’t stop here, as `ResNet-50` could only reach 68% accuracy at best for
a certain subject group, while it had between 19% and 31% accuracy for other groups.

- Parkinson

On the parkinson part, the results highly depend on the subjects taken for training. The best results involve the EEG
channels.
`RNNLGGnet` can occasionally reach 80% of accuracy when determining whether a
patient will develop the PD. However, this result is only attained for a certain subject group out of the 5
folds. For the other 4 groups, the accuracy can only reach 40% at best. Overall, the mean accuracy is only
about 50% while `ResNet-50` could only reach 68% accuracy at best for
a certain subject group, while it had between 19% and 31% accuracy for other groups.

Given the repartition of results, the labels should be more refined. The labels are currently naive for the parkinson
part. The whole REM segments would be labelled as `1` (meaning *will develop parkinson later*).

However, the manifestation of the parkinson's disease is probably not visible on the entirety of REM segments. It will
be better to isolate the characteristic marks.

Moreover, the time between the record time and the actual parkinson disease's contraction might be too long, thus not
having parkinson's disease's specific marks.

# Repository files

- Preprocessing
    - `concatenate_HOSP.py` concatenates `edf` files from the used dataset (not public). Can be useful as a base for
      other databases.
    - `edf_preprocessing.py` preprocesses `edf` files and store the results as `fif` files. Only useful for
      `visualize_fif.py`.
    - `edf_to_dat.py` preprocesses `edf` files and store results as `dat` files while distributing the files to ensure
      label distribution in the cross-validation groups. Used for training.
    - `explore.py` helps exploring the channels of each raw data file. Useful to verify which channels are in common.
    - `preprocess.py` contains the code to preprocess data.
- Training
    - `cross_validation.py` is in charge of the cross-validations. Based from `LGGnet`'s repository.
    - `main.py` is the code to run to train models. Based from `LGGnet`'s repository.
    - `model.py` defines the models (`LGGnet`, `RNNLGGnet`, `ResNet`). Based from `LGGnet`'s repository.
    - `prepare_data.py` prepares the data for cross-validations. Based from `LGGnet`'s repository.
    - `train.py` provides the functions used by `cross_validation.py` for training. Based from `LGGnet`'s repository.
    - `utils.py` has miscellaneous tools. Based from `LGGnet`'s repository.
- Checking/Visualising
    - `compare.py` evaluates the performances of multiple models.
    - `visualize.py` and `visualize_fif.py` plot `hdf` and `fif` files. Their purpose is not relevant enough as for now.
- Other
    - `example.py` gives an example usage code of a model trained using this repository.
- Sawtooth waves
    - `ConceFT.py` contains the code aiming to do the calculations
      from https://iopscience.iop.org/article/10.1088/1361-6579/ad66aa. It's purpose was to use it for detection of
      sawtooth waves but the execution time is too long in the current state. The parameters are therefore not tuned for
      sawtooth waves. Check the article for more details.
    - `stw_psd.py` was meant to find a way to detect sawtooth waves using PSD.

# What's next

RBD should be detectable using simple signal processing. I didn't have the time to clearly search for an efficient way.
But before searching such a way, I would advise to retrain `LGGnet` or `ResNet` on a single dataset containing subjects
suffering from RBD or not in order to be sure these models can effectively detect RBD.

Sawtooth waves seem to be the way to know whether a sufferer from RBD will detect a future Parkinson's disease. I am
imagining a check of the frequency of occurrences or the length of these waves. There are unfortunately not enough work
done for detection of these waves.