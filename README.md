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

Some of the best results are highlighted in green.
Given the repartition of results, the labels should be more refined. The labels are currently naive for the parkinson
part. The whole REM segments would be labelled as `1` (meaning *will develop parkinson later*).

However, the manifestation of the parkinson's disease is probably not visible on the entirety of REM segments. It will
be better to isolate the characteristic marks.

Moreover, the time between the record time and the actual parkinson disease's contraction might be too long, thus not
having parkinson's disease's specific marks.
