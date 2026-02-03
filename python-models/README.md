# Python Probes

The python probes perfectly replicate the mathematical operations of the burn probe. This means that it is possible to train a probe end-to-end with the encoder model to optimize performance.


Once the model and probe have finished training, the weights can be extracted from the probe model and imported into the rust runtime. Probe weights from python frameworks will produce identical outputs in rust.
