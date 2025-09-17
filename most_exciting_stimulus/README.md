# Using ViV1T to generate stimuli and test hypotheses

In this folder you can find the code to generate most-exciting stimuli using ViV1T (**Figure 5** and **Supplemental Figure 4**).
Note that we centre the stimuli to the estimated receptive field of the neuron. We use the model-estimated artificial receptive fields (aRF), see [tuning_retinotopy/README.md](../tuning_retinotopy/README.md).

Let us first look at some examples from the paper.
Here in particular, we will focus on contextual modulation at the neuron level.

### Natural and ViV1T-generated surrounds elicit stronger contextual modulation than gratings (Figure 5A)

<table style="width: 100%; table-layout: fixed;">
  <thead>
    <tr>
      <th rowspan="2" style="width: 1%; writing-mode: vertical-lr; white-space: nowrap;">Mouse</th>
      <th rowspan="2" style="width: 1%; writing-mode: vertical-lr; white-space: nowrap;">Neuron</th>
      <th rowspan="2" style="width: 24%;">Most<br>exciting<br>centre</th>
      <th colspan="3" style="width: 72%;">Most exciting surrounds</th>
    </tr>
    <tr>
      <th style="width: 24%;">Grating<br>video</th>
      <th style="width: 24%;">Natural<br>video</th>
      <th style="width: 24%;">Generated<br>video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>L</td>
      <td>050</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center.gif" alt="Grating center" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center_grating_video_surround.gif" alt="Grating video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseM_neuron050_grating_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%;"></td>
    </tr>
    <tr>
      <td>N</td>
      <td>054</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center.gif" alt="Grating center" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center_grating_video_surround.gif" alt="Grating video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_grating_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%;"></td>
    </tr>
    <tr>
      <td>N</td>
      <td>059</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center.gif" alt="Grating center" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center_grating_video_surround.gif" alt="Grating video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_grating_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%;"></td>
    </tr>
    <tr>
      <td>L</td>
      <td>071</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center.gif" alt="Grating center" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center_grating_video_surround.gif" alt="Grating video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_grating_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%;"></td>
    </tr>
  </tbody>
</table>

Quantifying the responses of many neurons to these stimuli, we found that natural surrounds typically allow for more excitation than grating surrounds,
and that ViV1T-generated surrounds allow for even more.

### Dynamic surrounds elicit stronger contextual modulation than static surrounds (Figure 5E)

<table style="width: 100%; table-layout: fixed;">
  <thead>
    <tr>
      <th rowspan="2" style="width: 1%; writing-mode: vertical-lr; white-space: nowrap;">Mouse</th>
      <th rowspan="2" style="width: 1%; writing-mode: vertical-lr; white-space: nowrap;">Neuron</th>
      <th rowspan="2" style="width: 18%;">Most<br>exciting<br>centre</th>
      <th colspan="4" style="width: 80%;">Most exciting surrounds</th>
    </tr>
    <tr>
      <th style="width: 18%;">Natural image</th>
      <th style="width: 18%;">Natural video</th>
      <th style="width: 18%;">Generated image</th>
      <th style="width: 18%;">Generated video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>L</td>
      <td>003</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center.gif" alt="Center" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_natural_image_surround.gif" alt="Natural image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_generated_image_surround.gif" alt="Generated image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron003_natural_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
    </tr>
    <tr>
      <td>N</td>
      <td>054</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center.gif" alt="Center" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_natural_image_surround.gif" alt="Natural image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_generated_image_surround.gif" alt="Generated image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron054_natural_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
    </tr>
    <tr>
      <td>N</td>
      <td>059</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center.gif" alt="Center" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_natural_image_surround.gif" alt="Natural image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_generated_image_surround.gif" alt="Generated image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseN_neuron059_natural_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
    </tr>
    <tr>
      <td>L</td>
      <td>071</td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center.gif" alt="Center" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_natural_image_surround.gif" alt="Natural image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_natural_video_surround.gif" alt="Natural video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_generated_image_surround.gif" alt="Generated image surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
      <td><img src="/figures/repo/most_exciting_stimuli/mouseL_neuron071_natural_center_generated_video_surround.gif" alt="Generated video surround" style="width: 100%; height: 100%; object-fit: contain;"></td>
    </tr>
  </tbody>
</table>
Quantifying the responses of many neurons to these stimuli, we found that dynamic surrounds typically allow for more excitation than static surrounds,
and that ViV1T-generated dynamic surrounds allow for more excitation than natural ones.

### Code
All of these were generated using the code in this folder:

- [estimate_neuron_reliability.py](estimate_neuron_reliability.py) estimate the neuron reliability before predicting and generating its MEIs and MEVs.
- [estimate_natural_stimuli_spectrum.py](estimate_natural_stimuli_spectrum.py) estimate the spatiotemporal power spectrum of the natural movies in the training set. This is needed to compute the KL divergence between generated stimuli and natural stimuli.
- single neuron most-exciting stimuli [single_neuron/](single_neuron)
  - most-exciting grating stimuli [single_neuron/grating_stimulus](single_neuron/grating_stimulus)
    - [predict_center_surround_gratings.py](single_neuron/grating_stimulus/predict_center_surround_gratings.py) predict centre-surround gratings to find the combination of centre and surround gratings that elicit the strongest response.
    [predict_natural_surround_with_grating_center.py](single_neuron/grating_stimulus/predict_natural_surround_with_grating_center.py) predict and find the natural video surround with the most-exciting grating centre fixed that elicits the strongest response.
  - most-exciting natural stimuli [single_neuron/natural_stimulus](single_neuron/natural_stimulus)
    - [predict_natural_center.py](single_neuron/natural_stimulus/predict_natural_center.py) predict and find the natural centre that elicits the strongest response.
    [predict_natural_surround.py](single_neuron/natural_stimulus/predict_natural_surround.py) predict and find the natural video surround with the most-exciting natural centre fixed that elicits the strongest response.
  - [generate_center_surround.py](single_neuron/generate_center_surround.py) generate the most-exciting image (MEI) and/or video (MEV) surround with the most-exciting grating/natural centre fixed.
- population most-exciting stimuli [population/](population) mirrors the single neuron folder structure but predict and generate stimuli to the population response.
