# Active-Learning-CXR-Classification
-   We compare
> ### Please read our [article](https://arxiv.org/pdf/2403.18871) for further information.

## Overview
-   Figure 1a shows the overview of our template guidance as a plug-and-play module for existing XAI methods.

## Code Usage
-   Supporting functions: `Classification_Functions.py` for classifier training, `Bootstrap_Functions.py` for calculating the standard devisation of model performance, `Uncertainty_Functions` for computing model uncertainty based on Monte Carlo simulations.
-   Initialization sample selection: `Diversity TXRV.py` is used for sample selection based on diversity and TXRV. `Random Sampling.py` is used for random sampling. Other files follows the same naming method.
-   Initialization model training: `Initialization Diversity TXRV.py` and `Initialization Diversity TXRV MLP3.py` is based on original pixels+VGG and TXRV representations+MLP3, respectively. Other files follows the same naming method.

## Citation
* Yuan, H., Zhu, M., Yang, R., ... & Hong, C. (2024). Clinical Domain Knowledge-Derived Template Improves Post Hoc AI Explanations in Pneumothorax Classification. arXiv preprint arXiv:2403.18871.
