# RNNLGGnet
An attempt to classify patients with RBD (REM sleep behaviour disorder) and predict whether an RBD patient will develop the Parkinson disease using an RNN-augmented LGGNet with PSG data.

The code architecture and the model is taken from `yi-ding-cs`'s repository (https://github.com/yi-ding-cs/LGG).
You probably want to check their repository for better understanding.

I added an RNN part (a Gru layer) and made the training in 3 phases.
- 1st phase: Train the non-RNN part (LGGNet) only
- 2nd phase: Train the RNN part only
- 3rd phase: Train the whole model