#plot_contaminations.py
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from make_ckov_prediction import  make_ckov_prediction_cut


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


def calculate_contamination_comp(all_dicts, momentum_bins, momentum_bin_centers):
    species_codes = [211, 321, 2212]  # PDG codes for pions, kaons, protons
    colors = {211: ('blue', 'lightblue'), 321: ('green', 'lightgreen'), 2212: ('red', 'pink')}
    markers = {211: 'o', 321: 's', 2212: 'D'}
    titles = ["Pion", "Kaon", "Proton"]
    methods = ["Standard HTM", "With Mass Hypothesis"]
    ckov_methods = ["TrackAttributes_ckovReconThisTrack", "TrackAttributes_ckovReconMassHypThisTrack"]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)

    for idx, true_species in enumerate(species_codes):
        ax = axes[idx]
        ax.set_title(f'Predicted species for true {titles[idx]}', fontsize=16)
        ax.set_xlabel('Momentum (GeV/c)')
        ax.set_ylabel('Count')

        for method_idx, ckov_method in enumerate(ckov_methods):
            pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
            momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
            ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

            contamination = {}
            for pred_species in species_codes:
                predicted_mask = ckov_recon == pred_species
                true_mask = pdg == true_species
                hist, _ = np.histogram(momentum[true_mask & predicted_mask], bins=momentum_bins)
                contamination[pred_species] = hist

            for pred_species in species_codes:
                ax.plot(momentum_bin_centers, contamination[pred_species],
                         label=f'{methods[method_idx]}: {titles[pred_species]}',
                         color=colors[pred_species][method_idx],
                         marker=markers[pred_species], linestyle='-', linewidth=2, markersize=8)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def calculate_contamination_norm_comp(all_dicts, momentum_bins, momentum_bin_centers):
    species_codes = [211, 321, 2212]  # PDG codes for pions, kaons, protons
    colors = {211: 'blue', 321: 'green', 2212: 'red'}
    markers = {211: 'o', 321: 's', 2212: 'D'}
    titles = ["Without Mass Hypothesis", "With Mass Hypothesis"]
    
    fig, axes = plt.subplots(1, len(species_codes), figsize=(24, 8), sharex=True, sharey=True)
    
    for idx, mass_hyp in enumerate([False, True]):
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack" if mass_hyp else "TrackAttributes_ckovReconThisTrack"
        pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
        momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
        ckov_recon = all_dicts["ThisTrack;1"][ckov_method]
        contamination = {}

        for true_species in species_codes:
            true_mask = pdg == true_species
            predicted_species = ckov_recon 
            contamination[true_species] = {}

            for pred_species in species_codes:
                predicted_mask = predicted_species == pred_species
                hist, _ = np.histogram(momentum[true_mask & predicted_mask], bins=momentum_bins)
                contamination[true_species][pred_species] = hist

        for i, true_species in enumerate(species_codes):
            ax = axes[i]
            total_counts = np.sum(list(contamination[true_species].values()), axis=0)
            ax.set_title(f'Normalized Prediction for {true_species} {titles[idx]}', fontsize=16)

            for pred_species in species_codes:
                normalized_counts = contamination[true_species][pred_species] / total_counts
                ax.plot(momentum_bin_centers, normalized_counts,
                        label=f'{titles[idx]}: {pred_species}', color=colors[pred_species],
                        marker=markers[pred_species], linestyle='-', linewidth=2, markersize=8)

            ax.set_xlabel('Momentum (GeV/c)')
            ax.set_ylabel('Normalized Count')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
# momentum_bins = np.linspace(start, end, num_bins)  # Define appropriately
# momentum_bin_centers = (momentum_bins[:-1] + momentum_bins[1:]) / 2
# calculate_contamination_norm_comnp(all_dicts, momentum_bins, momentum_bin_centers)


def calculate_contamination_norm(all_dicts, momentum_bins, momentum_bin_centers, mass_hyp = False):

    if mass_hyp:
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack"
        title = "With Mass Hypothesis"
    else:
        ckov_method = "TrackAttributes_ckovReconThisTrack"
        title = "Standard HTM"
    # Extract data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    th_pion = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"]
    th_kaon = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"]
    th_proton = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"]
    # TODO: remove those that are under threshold

    thresh_ckovs = [th_pion>0, th_pion>0, th_pion>0]

    differences = np.abs([ckov_recon - th_pion, ckov_recon - th_kaon, ckov_recon - th_proton])


    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)



    contamination = {}
    totals = {}
    species_codes = [211, 321, 2212]  # PDG codes for pions, kaons, protons

    # p_lim = m/(sqrt(n**2-1))

    p_lims = (0.4, 0.8, 1.2)
    # Compute the counts for each predicted species within the true species bins
    for true_species, thresh_ckov, p_lim in zip(species_codes, thresh_ckovs, p_lims):
        true_mask = (pdg == true_species) & (thresh_ckov)

        #momentum_bins = np.linspace(p_lim, 5, 50)  # Adjust the bins here

        total_hist, _ = np.histogram(momentum[true_mask], bins=momentum_bins)  # Total counts for normalization
        totals[true_species] = total_hist
        contamination_for_true = {}

        for pred_species, thresh_ckov in zip(species_codes, thresh_ckovs):
            predicted_mask = (predicted_species == pred_species)
            hist, _ = np.histogram(momentum[true_mask & (predicted_species == pred_species)], bins=momentum_bins)
            contamination_for_true[pred_species] = hist

        contamination[true_species] = contamination_for_true


    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}
    colors = {211: 'blue', 321: 'green', 2212: 'red'}
    markers = {211: 'o', 321: 's', 2212: 'D'}  # Different marker for each species

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True)
    # Plotting the normalized contamination for each true species
    for idx, true_species in enumerate([211, 321, 2212]):
        ax = axes[idx]

        if idx == 1:
            ax.set_title(f' {title}  \n Normalized Predicted Species for True {species_labels[true_species]}', fontsize=16)
        else :
            ax.set_title(f'Normalized Predicted Species for True {species_labels[true_species]}', fontsize=16)

        ax.set_xlabel('Momentum (GeV/c)')
        ax.set_ylabel('Normalized Count')

        total_counts = totals[true_species]

        for pred_species in [211, 321, 2212]:
            normalized_counts = contamination[true_species][pred_species] / total_counts
            ax.plot(momentum_bin_centers, normalized_counts,
                    label=f'Predicted {species_labels[pred_species]}',
                    color=colors[pred_species], marker=markers[pred_species], linestyle='-', linewidth=2, markersize=8)

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def calculate_contamination(all_dicts, momentum_bins, momentum_bin_centers, mass_hyp = False):


    if mass_hyp:
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack"
        title = "With Mass Hypothesis"
    else:
        ckov_method = "TrackAttributes_ckovReconThisTrack"
        title = "Standard HTM"
    # Extract data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    th_pion = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"]
    th_kaon = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"]
    th_proton = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"]

    # Determine predicted species using vectorized operations
    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)

    contamination = {}
    species_codes = [211, 321, 2212]  # PDG codes for pions, kaons, protons

    # Use numpy's histogramming functions to compute the counts for each predicted species within the true species bins
    for true_species in species_codes:
        contamination_for_true = {}
        true_mask = pdg == true_species
        for pred_species in species_codes:
            predicted_mask = predicted_species == pred_species
            hist, _ = np.histogram(momentum[true_mask & (predicted_species == pred_species)], bins=momentum_bins)
            contamination_for_true[pred_species] = hist
        contamination[true_species] = contamination_for_true


    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}
    colors = {211: 'blue', 321: 'green', 2212: 'red'}
    markers = {211: 'o', 321: 's', 2212: 'D'}  # Different marker for each species

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True)

    # Plotting the contamination for each true species
    for idx, true_species in enumerate([211, 321, 2212]):
        ax = axes[idx]

        if idx == 1:
            ax.set_title(f'{title} \n Predicted species for true {species_labels[true_species]}', fontsize = 16)
        else:
            ax.set_title(f'Predicted species for true {species_labels[true_species]}', fontsize = 16)
        
        ax.set_xlabel('Momentum (GeV/c)')
        ax.set_ylabel('Count')

        for pred_species in [211, 321, 2212]:
            ax.plot(momentum_bin_centers, contamination[true_species][pred_species],
                    label=f'Predicted {species_labels[pred_species]}',
                    color=colors[pred_species], marker=markers[pred_species], linestyle='-', linewidth=2, markersize=8)

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()





def calculate_purity_efficiency(all_dicts, momentum_bins, mass_hyp = False):


    if mass_hyp:
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack"
        title = "With Mass Hypothesis"
    else:
        ckov_method = "TrackAttributes_ckovReconThisTrack"
        title = "Standard HTM"

    # Extract the relevant data from all_dicts
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    th_pion = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"]
    th_kaon = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"]
    th_proton = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"]


    #p_lim = m/(sqrt(n2-1))

    # Determine predicted species using vectorized operations
    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)
    ######

    species_codes = [211, 321, 2212]  # PDG codes for pions, kaons, protons
    purity = {code: [] for code in species_codes}
    efficiency = {code: [] for code in species_codes}
    momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1]) / 2

    # Vectorized binning
    bins = np.digitize(momentum, bins=momentum_bins)

    # Vectorized calculation of true positives, false positives, and total true for each species
    for code in species_codes:
        # Convert boolean arrays to integers for use as weights in np.histogram
        true_positive_weights = (predicted_species[pdg == code] == code).astype(int)
        total_predicted_weights = (predicted_species == code).astype(int)

        true_positives, _ = np.histogram(momentum[pdg == code], bins=momentum_bins, weights=true_positive_weights)
        total_predicted, _ = np.histogram(momentum, bins=momentum_bins, weights=total_predicted_weights)
        total_true, _ = np.histogram(momentum[pdg == code], bins=momentum_bins)

        bin_purity = np.divide(true_positives, total_predicted, out=np.zeros_like(true_positives, dtype=float), where=total_predicted!=0)
        bin_efficiency = np.divide(true_positives, total_true, out=np.zeros_like(true_positives, dtype=float), where=total_true!=0)

        purity[code] = bin_purity
        efficiency[code] = bin_efficiency


    # Now let's plot the purity and efficiency for each particle type
    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}
    colors = {211: 'red', 321: 'blue', 2212: 'green'}

    plt.figure(figsize=(15, 6))

    # Purity plot
    plt.subplot(1, 2, 1)
    for code in species_labels:
        plt.plot(momentum_bin_centers, purity[code], label=species_labels[code], color=colors[code], marker='o')
    plt.title(f' Purity vs Momentum', fontsize =16)



    
    plt.xlabel('Momentum (GeV/c)', fontsize =16)
    plt.ylabel('Purity', fontsize =16)
    plt.legend()
    plt.grid()

    # Efficiency plot
    plt.subplot(1, 2, 2)
    for code in species_labels:
        plt.plot(momentum_bin_centers, efficiency[code], label=species_labels[code], color=colors[code], marker='o')
    plt.title('Efficiency vs Momentum', fontsize =16)
    plt.xlabel('Momentum (GeV/c)', fontsize =16)
    plt.ylabel('Efficiency', fontsize =16)
    plt.legend()
    plt.grid()
    plt.suptitle(title, fontsize=20, y=1.0005)  # Adjust the fontsize and position as needed

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def calculate_purity_efficiency_comp(all_dicts, momentum_bins):
    species_codes = [211, 321, 2212]  # PDG codes for pions, kaons, protons
    colors = {211: ('red', 'pink'), 321: ('blue', 'lightblue'), 2212: ('green', 'lightgreen')}
    markers = {211: 'o', 321: 's', 2212: 'D'}
    titles = ["Standard HTM", "With Mass Hypothesis"]
    methods = ["TrackAttributes_ckovReconThisTrack", "TrackAttributes_ckovReconMassHypThisTrack"]
    momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1]) / 2

    plt.figure(figsize=(15, 6))

    for i, method in enumerate(methods):
        purity = {code: [] for code in species_codes}
        efficiency = {code: [] for code in species_codes}

        for code in species_codes:
            pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
            momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
            ckov_recon = all_dicts["ThisTrack;1"][method]

            true_positive_weights = (ckov_recon == code).astype(int)
            total_predicted_weights = (ckov_recon == code).astype(int)

            true_positives, _ = np.histogram(momentum[pdg == code], bins=momentum_bins, weights=true_positive_weights)
            total_predicted, _ = np.histogram(momentum, bins=momentum_bins, weights=total_predicted_weights)
            total_true, _ = np.histogram(momentum[pdg == code], bins=momentum_bins)

            purity[code] = np.divide(true_positives, total_predicted, out=np.zeros_like(true_positives), where=total_predicted!=0)
            efficiency[code] = np.divide(true_positives, total_true, out=np.zeros_like(true_positives), where=total_true!=0)

        # Purity plot
        plt.subplot(1, 2, 1)
        for code in species_codes:
            plt.plot(momentum_bin_centers, purity[code], label=f'{titles[i]}: {species_codes[code]}',
                     color=colors[code][i], marker=markers[code])
        plt.title('Purity vs Momentum', fontsize=16)
        plt.xlabel('Momentum (GeV/c)', fontsize=16)
        plt.ylabel('Purity', fontsize=16)
        plt.legend()
        plt.grid(True)

        # Efficiency plot
        plt.subplot(1, 2, 2)
        for code in species_codes:
            plt.plot(momentum_bin_centers, efficiency[code], label=f'{titles[i]}: {species_codes[code]}',
                     color=colors[code][i], marker=markers[code])
        plt.title('Efficiency vs Momentum', fontsize=16)
        plt.xlabel('Momentum (GeV/c)', fontsize=16)
        plt.ylabel('Efficiency', fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
# momentum_bins = np.linspace(start, end, num_bins)  # Define appropriately
# calculate_purity_efficiency(all_dicts, momentum_bins)



