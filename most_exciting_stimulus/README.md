The `most_exciting_stimulus/` folder mostly consist of code to predict and generate most-exciting stimulus, images (MEIs) and videos (MEVs) using the movie-trained model (**Figure 5** and **Supplemental Figure 4**).

Note that you have to estimate the aRFs of the model prior to predicting and generating MEIs and MEVs. See [tuning_retinotopy/README.md](../tuning_retinotopy/README.md).

- [estimate_neuron_reliability.py](estimate_neuron_reliability.py) estimate the neuron reliability before predicting and generating its MEIs and MEVs.
- [estimate_natural_stimuli_spectrum.py](estimate_natural_stimuli_spectrum.py) estimate the spatiotemporal power spectrum of the natural movies in the training set. This is needed to compute the KL divergence between generated stimuli and natural stimuli.
- single neuron most-exciting stimuli [single_neuron/](single_neuron)
  - most-exciting grating stimuli [single_neuron/grating_stimulus](single_neuron/grating_stimulus)
    - [predict_center_surround_gratings.py](single_neuron/grating_stimulus/predict_center_surround_gratings.py) predict center-surround gratings to find the combination of center and surround gratings that elicit the strongest response.
    [predict_natural_surround_with_grating_center.py](single_neuron/grating_stimulus/predict_natural_surround_with_grating_center.py) predict and find the natural video surround with the most-exciting grating centre fixed that elicit the strongest response.
  - most-exciting natural stimuli [single_neuron/natural_stimulus](single_neuron/natural_stimulus)
    - [predict_natural_center.py](single_neuron/natural_stimulus/predict_natural_center.py) predict and find the natural center that elicit the strongest response.
    [predict_natural_surround.py](single_neuron/natural_stimulus/predict_natural_surround.py) predict and find the natural video surround with the most-exciting natural centre fixed that elicit the strongest response.
  - [generate_center_surround.py](single_neuron/generate_center_surround.py) generate the most-exciting image (MEI) and/or video (MEV) surround with the most-exciting grating/natural centre fixed.
- population most-exciting stimuli [population/](population) mirrors the single neuron folder structure but predict and generate stimuli to the population response.