# metadata for the Sensorium 2023 dataset

- [`statistics/`](statistics) contains additional metadata of the datasets, including the statistics of the stimuli, behavioral variables and responses.
- [`ood_features/`](ood_features) contains features of the OOD stimuli.
- [`tuning/`](tuning) contains the estimated tuning properties of each _in vivo_ and/or _in silico_ animal based on the OOD features.
- [`stimulus_ids.yaml`](stimulus_ids.csv) contains the stimulus ID for each trial in Sensorium 2023.
- [`video_ids.yaml`](video_ids.yaml) contains the video ID for each trial in Sensorium 2023.
## File structure
```
statistics/
  mouseA/
    behavior/
      min.npy
      max.npy
      mean.npy
      median.npy
      std.npy
     pupil_center/
     responses/
     videos/
  ...
ood_features/
  drifting_gabor/
    mouseB/
      2.npy
      8.npy
      ...
    ...
  gaussian_dots/
     mouseA/
        11.npy
        12.npy
        ...
     ...
tuning/
  mouseB/
    DSI.npy
    OSI.npy
  ...
stimulus_ids.csv
video_ids.csv
```

## OOD features
- Gaussian dots (`stimulus_id=2`)
  - Each `<trial_id>.npy` file contains the Gaussian dots features (x-axis center, y-axis center, radius, color) of the Gaussian dot in the trial. The color value can either be (0) white or (1) black.
- drifting gabor (`stimulus_id=4`)
  - Each `<trial_id>.npy` file contains the drifting gabor features (orientation, wavelength, frequency) of the trial.
