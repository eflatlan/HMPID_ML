
# Scatterplots
**make_ckov_prediction_cut_fig** in HMP_ML

#### Predicted specie kaon, probablity scatter plot per true specie
![kaonProbsScatter.jpg](kaonProbsScatter.jpg)

#### Predicted specie proton, probablity scatter plot per true specie
![protonProbsScatter.jpg](protonProbsScatter.jpg)



# Metrics comparison
**HMP_HTM.ipynb**
![standardHTMMetrics](standardHTMMetrics.png)

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


<div style="display: flex; justify-content: space-around;">
    <img src="ckov_recon.png" alt="ckov_recon" width="45%">
    <img src="ckov_reconmh.png" alt="ckov_reconmh" width="45%">
</div>


<div style="display: flex; justify-content: space-around;">
    <img src="diffStd.png" alt="diffStd.png" width="45%">
    <img src="diffmh.png" alt="diffmh" width="45%">
</div>

