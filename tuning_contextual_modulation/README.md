The `tuning_contextual_modulation/` folder mostly consists of code to estimate the centre-surround contextual modulation of the movie-trained models, mostly replicating the _in vivo_ experiments from [Keller et al. 2020](https://www.cell.com/neuron/fulltext/S0896-6273(20)30891-6), presented in **Figure 2** and **Figure 4**.

Note that you have to estimate the aRFs of the model prior to making predictions to the centre-surround gratings. See [tuning_retinotopy/README.md](../tuning_retinotopy/README.md).

- [predict_grating.py](predict_grating.py) inference the movie-trained model(s) to predict responses to centre, iso-surround and cross-surround gratings where the centre of the stimuli is adjusted based on the RF of each individual neuron (**Figure 2**).
- [predict_grating_population.py](predict_grating.py) inference the movie-trained model(s) to predict responses to centre, iso-surround and cross-surround gratings where the centre of the stimuli is adjusted based on the population RF (**Figure 4**).
- [estimate_contextual_modulation.py](estimate_contextual_modulation.py) estimate the centre-surround contextual modulation of the predicted responses.
- [visualize_contextual_modulation.py](visualize_contextual_modulation.py) visualize the center-surround contextual modulation from the predicted responses (**Figure 2**).
