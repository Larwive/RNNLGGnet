# RNNLGGnet

An attempt to classify patients with RBD (REM sleep behaviour disorder) and predict whether an RBD patient will develop
the Parkinson disease using an RNN-augmented LGGNet with PSG data.

The code architecture and the model is taken from `yi-ding-cs`'s repository (https://github.com/yi-ding-cs/LGG).
You probably want to check their repository for better understanding.

I added an RNN part (a Gru layer) and made the training in 3 phases.

- 1st phase: Train the non-RNN part (LGGNet) only
- 2nd phase: Train the RNN part only
- 3rd phase: Train the whole model

## 1st phase (LGG)

I added a sigmoid layer at the end of the `forward` method of the model and the `fc` layer is augmented by an
intermediate `Linear` layer.
It is trained exactly as in the original repository otherwise.

The data is shuffled between subjects while training.

## 2nd phase (RNN part only)

A bigger `RNNLGGNet` model is being trained.
The output of the `GCN` is used for the `GRU` layer and the `fc` layer from phase 1 (named `fcLGG`).
The output of the `GRU` layer goes into another `fc` layer.
Then, the outputs of the `fcLGG` and `fc` go into a `Linear` (named `out_weights`) layer before the `sigmoid`.

This phase only trains the `GRU`, `fc`, `fcLGG` and the final `Linear` layers.

Since the `GRU` layer's hidden state is intended to be specific to each subject, data isn't shuffled while training and linearly given to the model.
Then the hidden state is reset at each new subject.

## 3rd phase (entire model)

All the parameters are being trained simultaneously. Everything else is like the 2nd phase.