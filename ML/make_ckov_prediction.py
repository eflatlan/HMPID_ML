from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_ckov_prediction(all_dicts, ckov_method):
    """
    Make prediction of species based on HTM
    Either using standard HTM, or HTM with mass-hypothesis
    """
    # Extract the relevant data from all_dicts
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    th_pion = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"]
    th_kaon = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"]
    th_proton = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"]

    ref_index = 1.2904
    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938
    # angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    # angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    # angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    # Determining predicted species
    th_pion = np.arccos(np.sqrt(momentum**2 + mass_pion**2) / (momentum * ref_index))
    th_kaon = np.arccos(np.sqrt(momentum**2 + mass_kaon**2) / (momentum * ref_index))
    th_proton = np.arccos(np.sqrt(momentum**2 + mass_proton**2) / (momentum * ref_index))

    #p_lim = m/(sqrt(n2-1))

    # Create masks for valid thresholds
    valid_pion = ~(th_pion.isna() | (th_pion == 0))
    valid_kaon = ~(th_kaon.isna() | (th_kaon == 0))
    valid_proton = ~(th_proton.isna() | (th_proton == 0))

    # Calculate absolute differences, using np.inf where thresholds are invalid to ignore them in np.argmin
    differences = np.vstack([
        (ckov_recon - th).abs().where(valid, np.inf)  # Use .where to apply valid mask
        for valid, th in zip([valid_pion, valid_kaon, valid_proton], [th_pion, th_kaon, th_proton])
    ])

    # Determine indices of the minimum differences
    species_indices = np.argmin(differences, axis=0)

    # Map indices to species codes, default to np.nan if no valid species was found
    predicted_species = pd.Series(np.select(
        [species_indices == 0, species_indices == 1, species_indices == 2],
        [211, 321, 2212],
        default=np.nan
    ))


    return differences, predicted_species



def make_ckov_prediction_cut(all_dicts, ckov_method):
    
    """
    Make prediction of species based on HTM
    Either using standard HTM, or HTM with mass-hypothesis
    """
    
    
    # if adjusting the prediction based on abundance of species    
    abundance = {211: 0.8, 321: 0.1, 2212: 0.1}


    sigma = 0.01

    # Extract the relevant data from all_dicts
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    th_pion = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"]
    th_kaon = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"]
    th_proton = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"]


    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938
    # angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    # angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    # angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    # Determining predicted species
    #th_pion = np.arccos(np.sqrt(momentum**2 + mass_pion**2) / (momentum * 1.2904))
    #th_kaon = np.arccos(np.sqrt(momentum**2 + mass_kaon**2) / (momentum * 1.2904))
    #th_proton = np.arccos(np.sqrt(momentum**2 + mass_proton**2) / (momentum * 1.2904))


    #p_lim = m/(sqrt(n2-1))

    # Determine predicted species using vectorized operations

    z_pion = np.abs(ckov_recon - th_pion)/sigma
    z_kaon = np.abs(ckov_recon - th_kaon)/sigma
    z_proton = np.abs(ckov_recon - th_proton)/sigma

    # Calculating Gaussian amplitudes for the z-scores
    gaussian_amplitude_pion = norm.pdf(z_pion)
    gaussian_amplitude_kaon = norm.pdf(z_kaon)
    gaussian_amplitude_proton = norm.pdf(z_proton)

    # Scaling the Gaussian amplitudes by the relative abundance of the species
    # here it is not done, uncomment if using abundance
    gaussian_amplitude_pion# *= abundance[211]
    gaussian_amplitude_kaon# *= abundance[321]
    gaussian_amplitude_proton# *= abundance[2212]

    # Create masks for valid thresholds
    z_score_threshold = 3
    valid_pion = ~(th_pion.isna() | (th_pion == 0)) & (z_pion < z_score_threshold) & ~(z_pion.isna())  & (norm.pdf(z_pion) > 0.01)
    valid_kaon = ~(th_kaon.isna() | (th_kaon == 0)) & (z_kaon < z_score_threshold) & ~(z_kaon.isna())  & (norm.pdf(z_kaon) > 0.01)
    valid_proton = ~(th_proton.isna() | (th_proton == 0)) & (z_proton < z_score_threshold) & ~(z_proton.isna())  & (norm.pdf(z_proton) > 0.01)
    valid_species = valid_pion | valid_kaon | valid_proton

    # Calculate absolute differences, using np.inf where thresholds are invalid to ignore them in np.argmin
    
    diff_pion =  (ckov_recon - th_pion).abs().where(valid_pion, np.inf)
    diff_kaon =  (ckov_recon - th_kaon).abs().where(valid_kaon, np.inf)
    diff_proton =  (ckov_recon - th_proton).abs().where(valid_proton, np.inf)

    differences = np.vstack([
       diff_pion, diff_kaon, diff_proton
    ])


    adjusted_gaussian_amplitudes = np.array([
       gaussian_amplitude_pion,
       gaussian_amplitude_kaon,
       gaussian_amplitude_proton
    ])



    scaled_differences = differences / adjusted_gaussian_amplitudes
    species_indices = np.argmin(differences, axis=0)


    # Map indices to species codes, default to np.nan if no valid species was found
    predicted_species = pd.Series(np.select(
        [species_indices == 0, species_indices == 1, species_indices == 2],
        [211, 321, 2212],
        default=np.nan
    ))


    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}


    return differences, predicted_species


# also plot the zscore figures
def make_ckov_prediction_cut_fig(all_dicts, ckov_method):
        
    """
    Make prediction of species based on HTM
    Either using standard HTM, or HTM with mass-hypothesis
    Plot the figures
    """
    abundance = {211: 0.8, 321: 0.1, 2212: 0.1}


    sigma = 0.01

    # Extract the relevant data from all_dicts
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    th_pion = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"]
    th_kaon = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"]
    th_proton = all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"]


    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938
    # angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    # angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    # angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    # Determining predicted species
    #th_pion = np.arccos(np.sqrt(momentum**2 + mass_pion**2) / (momentum * 1.2904))
    #th_kaon = np.arccos(np.sqrt(momentum**2 + mass_kaon**2) / (momentum * 1.2904))
    #th_proton = np.arccos(np.sqrt(momentum**2 + mass_proton**2) / (momentum * 1.2904))


    #p_lim = m/(sqrt(n2-1))

    # Determine predicted species using vectorized operations

    z_pion = np.abs(ckov_recon - th_pion)/sigma
    z_kaon = np.abs(ckov_recon - th_kaon)/sigma
    z_proton = np.abs(ckov_recon - th_proton)/sigma

    # Calculating Gaussian amplitudes for the z-scores
    gaussian_amplitude_pion = norm.pdf(z_pion)
    gaussian_amplitude_kaon = norm.pdf(z_kaon)
    gaussian_amplitude_proton = norm.pdf(z_proton)

    # Scaling the Gaussian amplitudes by the relative abundance of the species
    #gaussian_amplitude_pion *= abundance[211]
    #gaussian_amplitude_kaon *= abundance[321]
    #gaussian_amplitude_proton *= abundance[2212]
    
    z_score_threshold = 3
    
    # Create masks for valid thresholds
    valid_pion = ~(th_pion.isna() | (th_pion == 0)) & (z_pion < z_score_threshold) & ~(z_pion.isna())#  & (norm.pdf(z_pion) > 0.01)
    valid_kaon = ~(th_kaon.isna() | (th_kaon == 0)) & (z_kaon < z_score_threshold)
    valid_proton = ~(th_proton.isna() | (th_proton == 0)) & (z_proton < z_score_threshold)
    valid_species = valid_pion | valid_kaon | valid_proton

    # Calculate absolute differences, using np.inf where thresholds are invalid to ignore them in np.argmin
    
    
    
    diff_pion =  (ckov_recon - th_pion).abs().where(valid_pion, np.inf)
    diff_kaon =  (ckov_recon - th_kaon).abs().where(valid_kaon, np.inf)
    diff_proton =  (ckov_recon - th_proton).abs().where(valid_proton, np.inf)

    differences = np.vstack([
       diff_pion, diff_kaon, diff_proton
    ])


    adjusted_gaussian_amplitudes = np.array([
       gaussian_amplitude_pion,
       gaussian_amplitude_kaon,
       gaussian_amplitude_proton
    ])


    scaled_differences = differences / adjusted_gaussian_amplitudes
    species_indices = np.argmin(scaled_differences, axis=0)

    # Map indices to species codes, default to np.nan if no valid species was found
    predicted_species = pd.Series(np.select(
        [species_indices == 0, species_indices == 1, species_indices == 2],
        [211, 321, 2212],
        default=np.nan
    ))

    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}

    # Outer loop over predicted species codes
    for j, pred_species_code in enumerate([211, 321, 2212]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # Inner loop over true species codes
        for i, (ax, species_code) in enumerate(zip(axes, [211, 321, 2212])):
            species_mask = (pdg == species_code)
            pred_species_mask = (predicted_species == pred_species_code) & species_mask
            z_species = np.abs(differences[i])  # Adjust index to match species code


            ax.scatter(momentum[pred_species_mask], ckov_recon[pred_species_mask], c=norm.pdf(z_species[pred_species_mask]), cmap='viridis', marker='o')
            ax.set_title(f'True: {species_labels[species_code]}')
            ax.set_xlabel('Momentum')
            ax.set_ylabel('CKOV Recon')
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1)
            ax.grid(True)

            ax.set_ylabel('Cherenkov Angle [rad]')

        
        
        axes[0].text(-0.15, 0.5, f'Predicted: {species_labels[pred_species_code]}', transform=axes[0].transAxes, ha='right', va='center', rotation=90, fontsize=14, color='red', backgroundcolor='white')



        cbar = plt.colorbar(ax.scatter([], [], c=[], cmap='viridis', marker='o'), ax=axes[-1], label='Z-score of Species')

        plt.suptitle(f' Z score of trues species')
        plt.tight_layout()
        plt.show()


    # Outer loop over predicted species codes
    for j, pred_species_code in enumerate([211, 321, 2212]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # Inner loop over true species codes
        for i, (ax, species_code) in enumerate(zip(axes, [211, 321, 2212])):
            species_mask = (pdg == species_code)
            pred_species_mask = (predicted_species == pred_species_code) & species_mask
            z_species = np.abs(differences[j])  # Adjust index to match species code



            ax.scatter(momentum[pred_species_mask], ckov_recon[pred_species_mask], c=norm.pdf(z_species[pred_species_mask]), cmap='viridis', marker='o')
            ax.set_title(f'True: {species_labels[species_code]}')
            ax.set_xlabel('Momentum')
            ax.set_ylabel('Cherenkov Angle [rad]')
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1)
            ax.grid(True)

        axes[0].set_ylabel('Cherenkov Angle [rad]')
        cbar = plt.colorbar(ax.scatter([], [], c=[], cmap='viridis', marker='o'), ax=axes[-1], label='Z-score of Species')
        axes[0].text(-0.15, 0.5, f'Predicted: {species_labels[pred_species_code]}', transform=axes[0].transAxes, ha='right', va='center', rotation=90, fontsize=14, color='red', backgroundcolor='white')
        plt.suptitle(f'Z score of predicted species')

        plt.tight_layout()
        plt.show()


    # Outer loop over predicted species codes
    for j, pred_species_code in enumerate([211, 321, 2212]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # Inner loop over true species codes
        for i, (ax, species_code) in enumerate(zip(axes, [211, 321, 2212])):
            species_mask = (pdg == species_code)
            pred_species_mask = (predicted_species == pred_species_code) & species_mask
            z_species = np.abs(differences[i])  # Adjust index to match species code



            ax.scatter(momentum[pred_species_mask], ckov_recon[pred_species_mask], c=norm.pdf(z_species[pred_species_mask]), cmap='viridis', marker='o')
            ax.set_title(f'True: {species_labels[species_code]}')
            ax.set_xlabel('Momentum')
            ax.set_ylabel('Cherenkov Angle [rad]')
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1)
            ax.grid(True)

        axes[0].set_ylabel('Cherenkov Angle [rad]')
        cbar = plt.colorbar(ax.scatter([], [], c=[], cmap='viridis', marker='o'), ax=axes[-1], label='Z-score of Species')

        plt.suptitle(f'Scatter Plot of probability for predicted specie')
        plt.tight_layout()
        plt.show()


    return differences, predicted_species