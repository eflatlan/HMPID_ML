# HMPID_ML.ipynb
[HMPID_ML.ipynb](ML/HMPID_ML.ipynb)

section **check specie probs** > plot_attributes_and_scale


## Data-distributions

### Track attributes

#### Track inclination : thetaP
![Track inclination : thetaP](ML/images/dataDists/thetAP.png)

#### MIP-cluster charge : qMip
![MIP-cluster charge : qMip](ML/images/dataDists/qMip.png)

#### Momentum
![Momentum](ML/images/dataDists/momentum.png)

### Cluster attributes
#### Cluster-photon angular resolution  : sigmaRing
![Cluster-photon angular resolution  : sigmaRing](ML/images/dataDists/sigmaRing.png)

#### Single-photon ADC charge distribution
![Single-photon ADC charge distribution](ML/images/dataDists/singlePhotQ.png)

#### Single-photon cluster-size distribution
![Single-photon cluster-siz distribution](ML/images/dataDists/singlePhotSize.png)

### Resolved cluster distributions

#### Raw Size distribution
![Raw Size distribution (number of pads fired in raw cluster)](ML/images/dataDists/rawSize.png)

#### Number of deconvoluted clusters
![Distribution of number of deconvoluted clusters per raw cluster (number of clusters forming raw cluster)](ML/images/dataDists/numRawClu.png)

### HTM distribution 
![Distribution of number of Hough Selected Photons under Mass-Hypothesis](ML/images/dataDists/numckovHough.png)


### Specie Z-score distributions

section **raw z-score**
plot_ckov_probs(all_dicts)

![Pion z-score](ML/images/dataDists/probTruePion.png)
![Kaon z-score](ML/images/dataDists/probTrueKaon.png)
![Proton z-score](ML/images/dataDists/probTrueProton.png)
