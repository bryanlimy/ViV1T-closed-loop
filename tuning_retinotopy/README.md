Estimate the artificial receptive fields (aRFs) of each _in silico_ neurons the models were trained on (**Figure 7** and Methods 4.5.1). The aRFs are used to estimate the centre of the receptive field for subsequent analyses.

- [predict_noise.py](predict_noise.py) inference the movie-trained model on full-field white noise images and save the predicted responses in `npz` files.
- [fit_aRFs.py](fit_aRFs.py) fit 2D Gaussian to the weighted average response to the white noise images to estimate its aRF.