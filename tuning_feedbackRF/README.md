The `tuning_feedbackRF/` folder mostly consist of code to estimate the feedback-dependent contextual modulation of the movie-trained models, mostly replicating the _in vivo_ experiments from [Keller et al. 2020](https://www.nature.com/articles/s41586-020-2319-4), presented in **Figure 3**.

Note that you have to estimate the aRFs of the model prior to making predictions to the centre-surround gratings. See [tuning_retinotopy/README.md](../tuning_retinotopy/README.md).

- [predict_grating.py](predict_grating.py) inference the movie-trained model(s) to predict response to classical and inverse gratings.
- [estimate_size_tuning_curve.py](estimate_size_tuning_curve.py) estimate the feedforward (ffRF) and feedback (fbRF) size tuning curves / receptive fields from the predicted responses to classical and inverse gratings.
- [visualize_size_tuning_curve.py](visualize_size_tuning_curve.py) visualize the ffRF and fbRF from the predicted responses (**Figure 3E**).
- [predict_onset.py](predict_onset.py) inference the movie-trained model(s) to predict response to classical and inverse gratings with 1000 repeats to estimate the response onset delay, as described in [Keller et al. 2020](https://www.nature.com/articles/s41586-020-2319-4).
- [visualize_response_onset.py](visualize_response_onset.py) visualize response onset delay to classical and inverse gratings (**Figure 3F**).