from scipy.stats import norm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


from make_ckov_prediction import make_ckov_prediction_cut, make_ckov_prediction, make_ckov_prediction_cut_fig

def plot_species_predictions(all_dicts, mass_hyp = False):


    if mass_hyp:
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack"
        title = "With Mass Hypothesis"
    else:
        ckov_method = "TrackAttributes_ckovReconThisTrack"
        title = "Standard HTM"

    # Extract relevant data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    # Theoretical Cherenkov angles
    p = np.linspace(0, 5, 500)
    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938
    angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    # Determining predicted species
    th_pion = np.arccos(np.sqrt(momentum**2 + mass_pion**2) / (momentum * 1.2904))
    th_kaon = np.arccos(np.sqrt(momentum**2 + mass_kaon**2) / (momentum * 1.2904))
    th_proton = np.arccos(np.sqrt(momentum**2 + mass_proton**2) / (momentum * 1.2904))

    # differences = np.vstack([np.abs(ckov_recon - th_pion), np.abs(ckov_recon - th_kaon), np.abs(ckov_recon - th_proton)])
    # species_indices = np.argmin(differences, axis=0)
    # predicted_species = np.select([species_indices == 0, species_indices == 1, species_indices == 2], [211, 321, 2212], default=np.nan)


    differences, predicted_species = make_ckov_prediction(all_dicts, ckov_method)

    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    species_codes = [211, 321, 2212]
    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}
    colors = {211: 'blue', 321: 'green', 2212: 'red'}
    theoretical_angles = [angle_pion, angle_kaon, angle_proton]
    plt.xlim(0, 5)  # Example limits for momentum in GeV/c
    plt.ylim(0, .85)  # Example limits for Cherenkov angle in degrees
    # Plotting each species
    for i, code in enumerate(species_codes):
        ax = axes[i]
        true_mask = (pdg == code)
        for pred_code in species_codes:
            mask = (predicted_species == pred_code) & true_mask
            axes[i].scatter(momentum[mask], ckov_recon[mask], alpha=0.5, label=f'Predicted: {species_labels[pred_code]}', color=colors[pred_code])
        
        
        axes[i].plot(p, theoretical_angles[i], 'k--', label=f'Theoretical {species_labels[code]} Angle')
        axes[i].set_title(f'True {species_labels[code]}')
        axes[i].set_xlabel('Momentum (GeV/c)')

        ax.set_xlim(0, 5)
        ax.set_ylim(0, 0.85)

        if i == 0:
            axes[i].set_ylabel('Cherenkov Angle [rad]')
        axes[i].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    fig = plt.gcf()  # Get the current figure
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Legger til en kantlinje (ramme) rundt hele figuren
    fig.patch.set_edgecolor('black')  # Farge på kantlinjen
    fig.patch.set_linewidth(2)        # Bredde på kantlinjen
    plt.show()



def plot_species_predictions_cut(all_dicts, mass_hyp = False):


    if mass_hyp:
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack"
        title = "With Mass Hypothesis"
    else:
        ckov_method = "TrackAttributes_ckovReconThisTrack"
        title = "Standard HTM"

    # Extract relevant data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    # Theoretical Cherenkov angles
    p = np.linspace(0, 5, 500)
    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938
    angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    # Determining predicted species
    th_pion = np.arccos(np.sqrt(momentum**2 + mass_pion**2) / (momentum * 1.2904))
    th_kaon = np.arccos(np.sqrt(momentum**2 + mass_kaon**2) / (momentum * 1.2904))
    th_proton = np.arccos(np.sqrt(momentum**2 + mass_proton**2) / (momentum * 1.2904))

    # differences = np.vstack([np.abs(ckov_recon - th_pion), np.abs(ckov_recon - th_kaon), np.abs(ckov_recon - th_proton)])
    # species_indices = np.argmin(differences, axis=0)
    # predicted_species = np.select([species_indices == 0, species_indices == 1, species_indices == 2], [211, 321, 2212], default=np.nan)


    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)

    # Plot setup
    

    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    species_codes = [211, 321, 2212]
    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}
    colors = {211: 'blue', 321: 'green', 2212: 'red'}
    theoretical_angles = [angle_pion, angle_kaon, angle_proton]
    plt.xlim(0, 5)  # Example limits for momentum in GeV/c
    plt.ylim(0, .85)  # Example limits for Cherenkov angle in degrees
    # Plotting each species¨

    th_values = [th_pion, th_kaon, th_proton]


    th_values = {211: angle_pion, 321: angle_kaon, 2212: angle_proton}

    for i, code in enumerate(species_codes):
        ax = axes[i]
        true_mask = (pdg == code)
        for i_pred, pred_code in enumerate(species_codes):
            
            mask = (predicted_species == pred_code) & true_mask


            z_species = np.abs(differences[i_pred])  # Adjust index to match species code


            mask2 = (mask) & (z_species < 3) & ( z_species > 0)
            axes[i].scatter(momentum[mask2], ckov_recon[mask2], alpha=0.5, label=f'Predicted: {species_labels[pred_code]}', color=colors[pred_code])
        

            th_val = th_values[pred_code]
            axes[i].plot(p, th_val  - 3 * 0.01, 'k--', label=f'Theoretical {species_labels[code]} Angle')
            axes[i].plot(p, th_val  + 3 * 0.01, 'k--', label=f'Theoretical {species_labels[code]} Angle')



        axes[i].plot(p, theoretical_angles[i], 'k-', label=f'Theoretical {species_labels[code]} Angle')



        if i == 1:
            axes[i].set_title(f' {title} \n True {species_labels[code]}')
        else:
            axes[i].set_title(f'True {species_labels[code]}')
        axes[i].set_xlabel('Momentum (GeV/c)')




        ax.set_xlim(0, 5)
        ax.set_ylim(0, 0.85)

        if i == 0:
            axes[i].set_ylabel('Cherenkov Angle [rad]')
        axes[i].legend()
    
    plt.tight_layout()
    fig = plt.gcf()  # Get the current figure
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Legger til en kantlinje (ramme) rundt hele figuren
    fig.patch.set_edgecolor('black')  # Farge på kantlinjen
    fig.patch.set_linewidth(2)        # Bredde på kantlinjen


    plt.show()



    # # plot the differences zscore
    # # Plot setup
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    # for i, code in enumerate(species_codes):
    #     ax = axes[i]
    #     true_mask = (pdg == code)
    #     # You would also need to retrieve or calculate the predictions and differences again
    #     # predictions, differences = make_ckov_prediction_cut(all_dicts, ckov_method, sigma)
    #     for pred_code in species_codes:
    #         mask = (predicted_species == pred_code) & true_mask
            
    #         # We will use the color intensity to represent the magnitude of the differences
    #         # Normalize the differences for plotting
    #         # We take the absolute value to avoid negative sizes and add 1 to avoid size 0
    #         normalized_differences = np.abs(differences[i][mask]) + 1
    #         # We can then map these differences to a suitable size for the scatter plot
    #         sizes = 50 * normalized_differences / normalized_differences.max()
            
    #         ax.scatter(momentum[mask], ckov_recon[mask], s=sizes, alpha=0.5,
    #                 label=f'Predicted: {species_labels[pred_code]}', color=colors[pred_code])

    #     # Plot the theoretical angles
    #     ax.plot(p, theoretical_angles[i], 'k--', label=f'Theoretical {species_labels[code]} Angle')
    #     ax.set_title(f'True {species_labels[code]}')
    #     ax.set_xlabel('Momentum (GeV/c)')
    #     ax.set_xlim(0, 5)
    #     ax.set_ylim(0, 0.85)

    #     if i == 0:
    #         ax.set_ylabel('Cherenkov Angle [rad]')
    #     ax.legend()
    # plt.show()










def plot_species_predictions_cut_ind(all_dicts, mass_hyp = False):


    if mass_hyp:
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack"
        title = "With Mass Hypothesis"
    else:
        ckov_method = "TrackAttributes_ckovReconThisTrack"
        title = "Standard HTM"

    # Extract relevant data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    # Theoretical Cherenkov angles
    p = np.linspace(0, 5, 500)
    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938
    angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    # Determining predicted species
    th_pion = np.arccos(np.sqrt(momentum**2 + mass_pion**2) / (momentum * 1.2904))
    th_kaon = np.arccos(np.sqrt(momentum**2 + mass_kaon**2) / (momentum * 1.2904))
    th_proton = np.arccos(np.sqrt(momentum**2 + mass_proton**2) / (momentum * 1.2904))

    # differences = np.vstack([np.abs(ckov_recon - th_pion), np.abs(ckov_recon - th_kaon), np.abs(ckov_recon - th_proton)])
    # species_indices = np.argmin(differences, axis=0)
    # predicted_species = np.select([species_indices == 0, species_indices == 1, species_indices == 2], [211, 321, 2212], default=np.nan)


    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)

    # Plot setup
    species_codes = [211, 321, 2212]
    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}
    colors = {211: 'blue', 321: 'green', 2212: 'red'}
    theoretical_angles = [angle_pion, angle_kaon, angle_proton]

    # Plotting each species¨

    th_values = [th_pion, th_kaon, th_proton]


    th_values = {211: angle_pion, 321: angle_kaon, 2212: angle_proton}



    # Outer loop over predicted species codes
    for j, pred_species_code in enumerate([211, 321, 2212]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # Inner loop over true species codes
        for i, (ax, species_code) in enumerate(zip(axes, [211, 321, 2212])):
            species_mask = (pdg == species_code)
            pred_species_mask = (predicted_species == pred_species_code) & species_mask
            z_species = np.abs(differences[j])  # Adjust index to match species code


            mask2 = (pred_species_mask) & (z_species < 3) & ( z_species > 0)
            ax.scatter(momentum[mask2], ckov_recon[mask2], alpha=0.5, label=f'Predicted: {species_labels[pred_species_code]}', color=colors[pred_species_code])
        

            th_val = th_values[species_code]
            ax.plot(p, th_val  - 3 * 0.01, 'k--', label=f'Theoretical {species_labels[species_code]} Angle')
            ax.plot(p, th_val  + 3 * 0.01, 'k--', label=f'Theoretical {species_labels[species_code]} Angle')

            ax.plot(p, th_val, 'k-', label=f'Theoretical {species_labels[pred_species_code]} Angle')
            ax.set_title(f'True: {species_labels[species_code]}')


            if i == 1:
                ax.set_title(f' {title} \n True {species_labels[species_code]}')
            else:
                ax.set_title(f'True {species_labels[species_code]}')

            ax.set_xlabel('Momentum (GeV/c)')
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 0.85)

            ax.set_ylabel('Cherenkov Angle [rad]')
            ax.legend()

        axes[0].text(-0.15, 0.5, f'Predicted: {species_labels[pred_species_code]}', transform=axes[0].transAxes, ha='right', va='center', rotation=90, fontsize=14, color='red', backgroundcolor='white')

        axes[0].set_xlabel('Momentum (GeV/c)')


 

    
        plt.tight_layout()
        fig = plt.gcf()  # Get the current figure
        
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Legger til en kantlinje (ramme) rundt hele figuren
        fig.patch.set_edgecolor('black')  # Farge på kantlinjen
        fig.patch.set_linewidth(2)        # Bredde på kantlinjen
    plt.show()








def plot_contamination(all_dicts, mass_hyp=False):
    if mass_hyp:
        ckov_method = "TrackAttributes_ckovReconMassHypThisTrack"
        title = "With Mass Hypothesis"
    else:
        ckov_method = "TrackAttributes_ckovReconThisTrack"
        title = "Standard HTM"

    # Extract relevant data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]

    # Theoretical Cherenkov angles
    p = np.linspace(0, 5, 500)
    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938
    angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    # Use prediction function to get predicted species and difference metrics
    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)

    # Species settings
    species_codes = [211, 321, 2212]
    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}

    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, (ax, species_code) in enumerate(zip(axes, species_codes)):
        # Masks for true species
        true_mask = (pdg == species_code)




        for i_pred, pred_code in enumerate(species_codes):



            pred_mask = (predicted_species == pred_code) & true_mask
            correct_id = (pred_code == species_code)
            color = 'green' if correct_id else 'red'  # Green if correctly identified, red if not

            z_species = np.abs(differences[i_pred])  # Adjust index to match species code


            mask2 = (pred_mask) & (z_species < 3) & ( z_species > 0)


            ax.scatter(momentum[mask2], ckov_recon[mask2], alpha=0.5, color=color, label=f'Predicted: {species_labels[pred_code]}' if correct_id else None)

        # Plot theoretical Cherenkov angle lines
        th_val = np.arccos(np.sqrt(p**2 + eval(f'mass_{species_labels[species_code].lower()}')**2) / (p * 1.2904))
        ax.plot(p, th_val, 'k-', label=f'Theoretical {species_labels[species_code]} Angle')
        ax.set_title(f'True: {species_labels[species_code]}')
        ax.set_xlabel('Momentum (GeV/c)')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 0.85)
        if i == 0:
            ax.set_ylabel('Cherenkov Angle [rad]')
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Legger til en kantlinje (ramme) rundt hele figuren
    fig.patch.set_edgecolor('black')  # Farge på kantlinjen
    fig.patch.set_linewidth(2)        # Bredde på kantlinjen    
    
    plt.show()
    
