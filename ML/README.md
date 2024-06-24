# HMPID_ML.ipynb
[HMPID_ML.ipynb](HMPID_ML.ipynb)

section **check specie probs** > plot_attributes_and_scale


## Data-distributions

### Track attributes

#### Track inclination : thetaP
![Track inclination : thetaP](images/dataDists/thetAP.png)

#### MIP-cluster charge : qMip
![MIP-cluster charge : qMip](images/dataDists/qMip.png)

#### Momentum
![Momentum](images/dataDists/momentum.png)

### Cluster attributes
#### Cluster-photon angular resolution  : sigmaRing
![Cluster-photon angular resolution  : sigmaRing](images/dataDists/sigmaRing.png)

#### Single-photon ADC charge distribution
![Single-photon ADC charge distribution](images/dataDists/singlePhotQ.png)

#### Single-photon cluster-size distribution
![Single-photon cluster-siz distribution](images/dataDists/singlePhotSize.png)

### Resolved cluster distributions

#### Raw Size distribution
![Raw Size distribution (number of pads fired in raw cluster)](images/dataDists/rawSize.png)

#### Number of deconvoluted clusters
![Distribution of number of deconvoluted clusters per raw cluster (number of clusters forming raw cluster)](images/dataDists/numRawClu.png)

### HTM distribution 
![Distribution of number of Hough Selected Photons under Mass-Hypothesis](images/dataDists/numckovHough.png)


### Specie Z-score distributions

section **raw z-score**
plot_ckov_probs(all_dicts)

#### True particle is pion

![Pion z-score](images/dataDists/probTruePion.png)

The z-score (number of std-deviations away from each specie is horizontally)


#### True particle is kaon

![Kaon z-score](images/dataDists/probTrueKaon.png)

The z-score (number of std-deviations away from each specie is horizontally)


#### True particle is proton

![Proton z-score](images/dataDists/probTrueProton.png)

The z-score (number of std-deviations away from each specie is horizontally)


**The z-score per specie**

# Split distributions 

**To illustrate that the shuffling is done correctly i.e., we shuffle it such that momentum-distribution is similiar, distrribution of species is similar**

**also to illustrate that they are shuffled with the indices; not all pions coming first fx**

![Proton z-score](images/SplitPlots/countSpecies.jpg)

![Proton z-score](images/SplitPlots/momentumPerSpecie.png)

![Proton z-score](images/SplitPlots/momentums2.png)

![Proton z-score](images/SplitPlots/momentum.png)


# Output plots 

**Just example, Model was not tuned, and we did not have adequate data**

![Proton z-score](images/OutputPlots/PR.png)

![Proton z-score](images/OutputPlots/TrainValid.png)

![Proton z-score](images/OutputPlots/confMatrix.png)

![Proton z-score](images/OutputPlots/momentumAccuracyML.png)

![Proton z-score](images/OutputPlots/proportionPredictedML.png)
