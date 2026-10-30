# ThunderBooks
Jupyter notebooks on synchrotron radiation and radio transients
Containing tutorials and code snippets for understanding your results


**Basics** notebook requires the diagrams in `output/figures/`
This notebook may be all you need if you want the most basic understanding of synchrotron emission and how the first models
(e.g. van der Laan) are calculated

**Equipartition analysis** provides the full derivation of the equipartition argument, with a more precise formula than the approximation provided in *Basics*

Many thanks to Miriam Nyamai, Joe Bright, Lauren Rhodes and Sara Motta for test-driving and minor edits. Additional test-drivers very welcome.

## Plotting and animations in Basics

Run `Basics.ipynb` with Python 3.10 or later from this repository folder.
Install its dependencies with `python -m pip install -r requirements.txt`. Physical constants,
model inputs, and all synchrotron equations stay in the notebook;
`synchrotron_plotting.py` contains the plotting and animation functions.
Keep that file alongside the notebook when copying it elsewhere.

The animated-spectrum section uses the model arrays calculated immediately above
it. It traces the same flux measurements on linear-linear and log-log axes,
alongside the evolving spectrum. A separate figure shows the normalised field,
electron distribution, magnetic flux and reference-electron frequency. A completed static light-curve figure is retained
immediately below, along with the earlier static spectrum figures.

Inline playback uses a GIF, so it needs neither JavaScript nor a video encoder.
The saved PNG fallback remains visible in viewers that cannot display animations.
For play/pause and a time slider, open
`output/graphs/evolving_synchrotron.html` in a browser. The GIF and browser
player are self-contained and work offline. Rerun the cells after changing model
parameters. Some editors require trusting the notebook or rerunning its output
cells before showing rich HTML; the GIF can also be opened directly.

## Notes and generated files

- `output/figures/`: explanatory diagrams and reference images.
- `output/graphs/`: plotted figures, SVGs, GIF animations, and controlled HTML players.
- `scripts/`: reproducible standalone scripts for the labelled note graphs.

Playback begins at 1 second. The spectrum's frequency grid includes the early
high-frequency turnover, and the linear light-curve view includes an early-time
inset. The HTML player provides playback controls alongside the saved GIF.
