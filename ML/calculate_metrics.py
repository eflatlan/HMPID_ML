
from scipy.stats import norm

def filter_data_by_zscore_and_momentum(all_dicts, z_score_threshold, momentum_range, ckov_method):
    
    """
    Filter data based on 
        *   z-score (num std-dev from theoretical value)
        *   momentum-range
    
    
    
    Args:
    all_dicts : copy of dict with data sent
    z_score_threshold : number of std-dev to set the threshold 
    momentum_range : the range of momentum to consider
    ckov_method : string, specifying to use either standard HTM or HTM with mass-hypothesis
    """
    
    # Extract relevant data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]
    
    # Get predictions and differences
    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)
    
    # Apply filters
    z_score_mask = np.abs(differences) < z_score_threshold
    momentum_mask = (momentum >= momentum_range[0]) & (momentum <= momentum_range[1])
    mask = z_score_mask & momentum_mask
    
    # Filter the data
    filtered_momentum = momentum[mask]
    filtered_ckov_recon = ckov_recon[mask]
    filtered_pdg = pdg[mask]
    filtered_predicted_species = predicted_species[mask]
    
    return filtered_momentum, filtered_ckov_recon, filtered_pdg, filtered_predicted_species

def plot_histograms_with_gaussians(data, species_labels, bins=30):
    """
    Unpacks the input data
    Plots the data with gaussian fits
    """
    momentum, ckov_recon, pdg, predicted_species = data
    
    fig, ax = plt.subplots()
    
    for code in np.unique(predicted_species):
        species_data = ckov_recon[predicted_species == code]
        ax.hist(species_data, bins=bins, alpha=0.5, label=f'{species_labels[code]} Predicted', density=True)
        
        # Fit and plot Gaussian
        mu, std = norm.fit(species_data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k--', linewidth=2)
    
    ax.set_xlabel('Cherenkov Angle [rad]')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()
    
def plot_species_specific_histograms(data, species_codes, species_labels):
    momentum, ckov_recon, pdg, predicted_species = data
    
    for true_species in species_codes:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, pred_species in enumerate(species_codes):
            ax = axes[i]
            mask = (pdg == true_species) & (predicted_species == pred_species)
            species_data = ckov_recon[mask]
            ax.hist(species_data, bins=30, alpha=0.5, label=f'{species_labels[pred_species]} Predicted', density=True)
            
            # Fit and plot Gaussian
            mu, std = norm.fit(species_data)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k--', linewidth=2)
            
            ax.set_title(f'True {species_labels[true_species]}, Predicted {species_labels[pred_species]}')
            ax.set_xlabel('Cherenkov Angle [rad]')
            ax.set_ylabel('Density')
            ax.legend()
        
        plt.suptitle(f'True {species_labels[true_species]} Histograms')
        plt.tight_layout()
        plt.show()
