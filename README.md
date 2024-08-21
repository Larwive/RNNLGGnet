# RNNLGGnet

An attempt to classify patients with RBD (REM sleep behaviour disorder) and predict whether an RBD patient will develop
the Parkinson disease using an RNN-augmented LGGNet with PSG data.

The code architecture and the model is taken from `yi-ding-cs`'s repository (https://github.com/yi-ding-cs/LGG).
You probably want to check their repository for better understanding.

I added an RNN part (a Gru layer) and made the training in 3 phases.

- 1st phase: Train the non-RNN part (LGGNet) only
- 2nd phase: Train the RNN part only
- 3rd phase: Train the whole model

A resnet50-inspired model is also used to compare with `RNNLGGnet`. This model is adapted to take the same inputs as
`RNNLGGnet`.

## 1st phase (LGG)

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

# Results

Don't train phases 2 and 3. Phase 1 is largely better (which actually consists of a slightly less overfitting `LGG`).

Link to results: https://docs.google.com/document/d/1E_bEYQ98wCeGFsEgza_1uB-pqniT3MLgYNjN5CrfDJs

**Note that the `enhanced` mention means that the phase 1 is the modified one.**

- RBD

The results on classifying whether a patient has RBD or not is very promising, reaching 99.98% accuracy at best on
patients the model has never seen. Globally, phase 2 sharply decreases the accuracy by privileging one output over the
other. Phase 3 might fix phase 2 a little or even make the predictions worse. So only use phase 1.

`resnet` has roughly the same results as `RNNLGGnet` (actually `LGGnet`).

- Parkinson

On the parkinson part, the results highly depend on the subjects taken for training. The best results involve the EEG
channels.

Some of the best results are highlighted in green.
Given the repartition of results, the labels should be more refined. The labels are currently naive for the parkinson
part. The whole REM segments would be labelled as `1` (meaning *will develop parkinson later*).

However, the manifestation of the parkinson's disease is probably not visible on the entirety of REM segments. It will
be better to isolate the characteristic marks.

Moreover, the time between the record time and the actual parkinson disease's contraction might be too long, thus not
having parkinson's disease's specific marks.
