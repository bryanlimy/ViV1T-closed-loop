# Using ViV1T to generate stimuli and test hypotheses

In this folder you can find the code to generate most-exciting stimuli using ViV1T (Figure 5 and Supplemental Figure 4).
Note that we typically center the generated stimuli on the receptive field of the neuron, thus you need to have those extracted somehow.
We use the model-estimated receptive fields (aRF), see [tuning_retinotopy/README.md](../tuning_retinotopy/README.md).

Let us first look at some examples from the paper.
Here in particular, we will focus on contextual modulation at the neuron level.

## Natural and ViV1T-generated surrounds elicit stronger contextual modulation than gratings (Figure 5A)

<table>
  <thead>
    <tr>
      <th rowspan="2">Most exciting center</th>
      <th colspan="3">Most exciting surrounds</th>
    </tr>
    <tr>
      <th>Grating video</th>
      <th>Natural video</th>
      <th>Generated video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center_grating_video_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center_grating_video_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center_grating_video_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center_grating_video_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
  </tbody>
</table>

## Dynamic surrounds elicit stronger contextual modulation than static surrounds (Figure 5E)

<table>
  <thead>
    <tr>
      <th rowspan="2">Most exciting center</th>
      <th colspan="4">Most exciting surrounds</th>
    </tr>
    <tr>
      <th>Natural image</th>
      <th>Natural video</th>
      <th>Generated image</th>
      <th>Generated video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_natural_image_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_generated_image_surround.gif" alt="Generated video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_natural_image_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_generated_image_surround.gif" alt="Generated video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_natural_image_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_generated_image_surround.gif" alt="Generated video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
    <tr>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center.gif" alt="Grating center"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_natural_image_surround.gif" alt="Grating video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_natural_video_surround.gif" alt="Natural video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_generated_image_surround.gif" alt="Generated video surround"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_generated_video_surround.gif" alt="Generated video surround"></td>
    </tr>
  </tbody>
</table>


## Code
All of these were generated using the code in this folder:

- [estimate_neuron_reliability.py](estimate_neuron_reliability.py) estimate the neuron reliability before predicting and generating its MEIs and MEVs.
- [estimate_natural_stimuli_spectrum.py](estimate_natural_stimuli_spectrum.py) estimate the spatiotemporal power spectrum of the natural movies in the training set. This is needed to compute the KL divergence between generated stimuli and natural stimuli.
- single neuron most-exciting stimuli [single_neuron/](single_neuron)
  - most-exciting grating stimuli [single_neuron/grating_stimulus](single_neuron/grating_stimulus)
    - [predict_center_surround_gratings.py](single_neuron/grating_stimulus/predict_center_surround_gratings.py) predict center-surround gratings to find the combination of center and surround gratings that elicit the strongest response.
    [predict_natural_surround_with_grating_center.py](single_neuron/grating_stimulus/predict_natural_surround_with_grating_center.py) predict and find the natural video surround with the most-exciting grating centre fixed that elicits the strongest response.
  - most-exciting natural stimuli [single_neuron/natural_stimulus](single_neuron/natural_stimulus)
    - [predict_natural_center.py](single_neuron/natural_stimulus/predict_natural_center.py) predict and find the natural center that elicits the strongest response.
    [predict_natural_surround.py](single_neuron/natural_stimulus/predict_natural_surround.py) predict and find the natural video surround with the most-exciting natural centre fixed that elicits the strongest response.
  - [generate_center_surround.py](single_neuron/generate_center_surround.py) generate the most-exciting image (MEI) and/or video (MEV) surround with the most-exciting grating/natural centre fixed.
- population most-exciting stimuli [population/](population) mirrors the single neuron folder structure but predict and generate stimuli to the population response.
