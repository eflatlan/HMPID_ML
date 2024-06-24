
# Scatterplots
**make_ckov_prediction_cut_fig** in [HMPID_ML.ipynb](../ML/HMPID_ML.ipynb)

#### Predicted specie kaon 

![kaonProbsScatter.jpg](kaonProbsScatter.jpg)

The figure shows the probablity scatter plot per true specie


#### Predicted specie proton

![protonProbsScatter.jpg](protonProbsScatter.jpg)

The figure shows the probablity scatter plot per true specie


# Metrics comparison
**HMP_HTM.ipynb**


![standardHTMMetrics](standardHTMMetrics.png)

**Efficiciency and Purity vs momentum**


![masshypMetrics](masshypMetrics.png)

## With masshypothesis :

Change by changning from **TrackAttributes_ckovReconMassHypThisTrack** to **TrackAttributes_ckovReconThisTrack** for **ckov_recon** in **calculate_contamination** / **calculate_purity_efficiency**

### Section Contamination

![predictedSpecie](predictedSpecie.png)

### Section Normalised

![predictedSpecieNorm](predictedSpecieNorm.png)

### Section "Efficiciency and Purity vs momentum"

![purityMomentumHTM](purityMomentumHTM.png)



### Section Plot Ckov photons histograms

![purityMomentumHTM](thetaCer.png)

Reconstructed Cherenkov angle with and without MH
<div style="display: flex; justify-content: space-around;">
    <img src="ckov_recon.png" alt="ckov_recon" width="45%">
    <img src="ckov_reconmh.png" alt="ckov_reconmh" width="45%">
</div>


Difference between theoretical values and reconstruted values:

<div style="display: flex; justify-content: space-around;">
    <img src="diffStd.png" alt="diffStd.png" width="45%">
    <img src="diffmh.png" alt="diffmh" width="45%">
</div>

