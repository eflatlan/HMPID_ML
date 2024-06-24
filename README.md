# HMPID_ML
ML and HTM repo for HMP

## Data-generation
Use branch [HMPID_MC_TO_ROOT/massHyp](https://github.com/eflatlan/HMPID_MC_TO_ROOT/tree/massHyp) to simulate data.
The **massHyp** branch has the correct data fields for compatibility with the notebooks here.
The notebooks converts TTRee to Numpy-dictionaries and Pandas Dataframes

## Content 
### STATS_ROOT
[STATS_ROOT_python3_10.ipynb](STATS_ROOT_python3_10.ipynb)

Notebook allowing the use of ROOT in Python,
Some examples of using ROOT histograms and Gaussian KDE for MIP charge of 2 species compared. 

### HTM 
#### 
[HTM/HMP_HTM.ipynb](HTM/HMP_HTM.ipynb)

Comparing standard HTM and mass-hypothesis HTM.
Some plots comparing Ckov values
Also plots of contamination, efficiency, purity etc.



### Statisitc
[Stats/HMPStats.ipynb](Stats/HMPStats.ipynb)

Plots of miscallenous things
- Scatterplot of track-inclination vs momentum
- Number of Cherenkov photons vs momentum
-  Number of Cherenkov photons vs ThetaC
-  Number of Cherenkov photons vs sin2(ThetaC)


### ML
[ML/HMPID_ML.ipynb](ML/HMPID_ML.ipynb)
Scaling, building model etc

- Train-dev-test split
- Efficiency, purity as function of momentum
- Accuracy as function of momentum
- P-R Curves
- Train-loss curves


#### Helper-functions
##### To use notebooks using these, one must specify the paths
- [ML/histograms.py](ML/histograms.py)
- [ML/make_ckov_prediction.py](ML/make_ckov_prediction.py)
- [ML/plot_contaminations.py](ML/plot_contaminations.py)
- [ML/read_file_and_pad.py](ML/read_file_and_pad.py)
- [ML/map_helper_functions.py](ML/map_helper_functions.py)
- [ML/make_ckov_prediction.py](ML/make_ckov_prediction.py)
- [ML/ParticleObjectProcessor.py](ML/ParticleObjectProcessor.py)
- [ML/make_test_data.py](ML/make_test_data.py)
- [ML/make_ckov_prediction.py](ML/make_ckov_prediction.py)
- [ML/ParticleObject.py](ML/ParticleObject.py)
- [ML/plot_species_predictions.py](ML/plot_species_predictions.py)
- [ML/calculate_metrics.py](ML/calculate_metrics.py)
- [ML/import_helpers.py](ML/import_helpers.py)
- [ML/scatter_plots.py](ML/scatter_plots.py)
- [ML/confusion_matrix.py](ML/confusion_matrix.py)


