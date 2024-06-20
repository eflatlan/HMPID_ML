from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from make_ckov_prediction import make_ckov_prediction_cut




species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}

from import_helpers import *
import matplotlib
#matplotlib.use('TkAgg')  # Use in standalone scripts to specify an interactive backend
def plot_numerical_histogram(df, column_name, output_file, **kwargs):
    """
    Plots a numerical histogram for a specified column in a DataFrame and saves it as a file.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to plot.
    output_file (str): File path where the plot will be saved.
    bins (int): Number of bins to use in the histogram.
    """
    if column_name in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column_name].dropna(), alpha=0.75, color='blue', **kwargs)  # Drop NaN values
        plt.title(f'Numerical Histogram of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(True)
        fig = plt.gcf()  # Get the current figure

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Legger til en kantlinje (ramme) rundt hele figuren
        fig.patch.set_edgecolor('black')  # Farge på kantlinjen
        fig.patch.set_linewidth(2)        # Bredde på kantlinjen        
        
        plt.show() 

        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved to {output_file}")
    else:
        print(f"Column {column_name} does not exist in the DataFrame.")


def plot_histogram(df, column_name, output_file):
    """
    Plots a categorical histogram for specified PDG codes in a DataFrame column,
    treating all values as their absolute values, and saves the plot to a file.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to plot.
    output_file (str): Filename to save the plot.
    """
    if column_name in df.columns:
        pdg_codes_of_interest = [11, 13, 211, 321, 2212, 22]
        # Filter and take absolute values
        filtered_series = df[column_name].abs()
        filtered_series = filtered_series[filtered_series.isin(pdg_codes_of_interest)]

        # Convert the series to categorical with the specified categories
        filtered_series = pd.Categorical(filtered_series, categories=pdg_codes_of_interest, ordered=True)

        # Plotting
        plt.figure(figsize=(10, 6))
        pd.Series(filtered_series).value_counts().sort_index().plot(kind='bar', color='blue', alpha=0.75)
        plt.title(f'Categorical Histogram of {column_name}')
        plt.xlabel('Absolute PDG Code')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.xticks(rotation=0)  
        fig = plt.gcf()

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Legger til en kantlinje (ramme) rundt hele figuren
        fig.patch.set_edgecolor('black')  # Farge på kantlinjen
        fig.patch.set_linewidth(2)        # Bredde på kantlinjen        
        
        plt.show()
        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved to {output_file}")
    else:
        print(f"Column {column_name} does not exist in the DataFrame.")


# Example usage:
# plot_histogram(df, 'mcTruth_pdgCodeClu')


def calculate_distances(df):
    """ 
    Calculates eucl. distance from MIP to extrapolated PC impact
    Args : dict holding the MIP and PC (x, y)
    """
    x_mip = df['TrackAttributes_xMipThisTrack']
    x_pc = df['TrackAttributes_xPCThisTrack']
    y_mip = df['TrackAttributes_yMipThisTrack']
    y_pc = df['TrackAttributes_yPCThisTrack']

    # Calculate the Euclidean distance using numpy for vectorized operations
    dist = np.sqrt((x_mip - x_pc)**2 + (y_mip - y_pc)**2)
    return dist


def plot_distance_histogram(df, output_file, bins=30, range = (-12, 12), title = "placeholder"):
    """
    Calculates the Euclidean distance between mip and PC points and plots a histogram.

    Args:
    df (pandas.DataFrame): DataFrame containing the mip and PC coordinates.
    output_file (str): Path to save the histogram image.
    bins (int): Number of bins in the histogram.
    """
    # Calculate distances
    df['TrackAttributes_mipPcDistThisTrack'] = calculate_distances(df)
    mask = (df['TrackAttributes_mipPcDistThisTrack'] >= range[0]) & (df['TrackAttributes_mipPcDistThisTrack'] <= range[1])
    # Plotting the histogram of distances
    plt.figure(figsize=(10, 6))
    dist_values = df.loc[mask, 'TrackAttributes_mipPcDistThisTrack']

    mean_dist = np.mean(dist_values)
    std_dist = np.std(dist_values)


    plt.hist(df['TrackAttributes_mipPcDistThisTrack'].dropna(), bins=bins, alpha=0.75, color='blue', range = (0, range[1]))
    plt.title(f'{title} Distance (mip-PC)')
    plt.xlabel('Distance [cm]')
    plt.ylabel('Frequency')
    plt.annotate(f'Mean (µ): {mean_dist:.2f}\nStd Dev (σ): {std_dist:.2f}', xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)
    plt.grid(True)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig = plt.gcf()  

    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('black')
    plt.show()
    plt.savefig(output_file)

    plt.close()

    print(f"Plot saved to {output_file}")

# Example usage:
# plot_distance_histogram(combined_df, 'mip_pc_distance_histogram.png')

def add_component_distances(df):

    df['TrackAttributes_xMipPcDistThisTrack'] = df['TrackAttributes_xMipThisTrack'] - df['TrackAttributes_xPCThisTrack']
    df['TrackAttributes_yMipPcDistThisTrack'] = df['TrackAttributes_yMipThisTrack'] - df['TrackAttributes_yPCThisTrack']
    return df



def plot_component_distance_histograms(df, output_file_prefix, bins=30, range = (-12, 12), title = "placeholder"):
    """
    Plots separate histograms for the x and y distances between mip and PC.

    Args:
    df (pandas.DataFrame): DataFrame containing the calculated distances.
    output_file_prefix (str): Base path to save histogram images, appended with '_x' and '_y' for respective histograms.
    bins (int): Number of bins in the histogram.
    """
    # Check if distance calculations are done
    if 'TrackAttributes_xMipPcDistThisTrack' not in df or 'TrackAttributes_yMipPcDistThisTrack' not in df:
        df = add_component_distances(df)

    mask_x = (df['TrackAttributes_xMipPcDistThisTrack'] >=  range[0]) & (df['TrackAttributes_xMipPcDistThisTrack'] <=  range[1])
    mask_y = (df['TrackAttributes_yMipPcDistThisTrack'] >=  range[0]) & (df['TrackAttributes_yMipPcDistThisTrack'] <=  range[1])

    x_values = df.loc[mask_x, 'TrackAttributes_xMipPcDistThisTrack']
    y_values = df.loc[mask_y, 'TrackAttributes_yMipPcDistThisTrack']

    mean_x = np.mean(x_values)
    std_x = np.std(x_values)

    mean_y = np.mean(y_values)
    std_y = np.std(y_values)

    # Plotting the histogram of x distances
    plt.figure(figsize=(10, 6))
    plt.hist(x_values, bins=bins, alpha=0.75, color='blue', label='x Distance (mip-PC)', range=range)
    plt.title(f'{title} x Component Distances (mip-PC)')
    plt.xlabel('x Distance')
    plt.ylabel('Frequency')
    plt.annotate(f'Mean (µ): {mean_x:.2f}\nStd Dev (σ): {std_x:.2f}', xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)
    plt.grid(True)
    fig = plt.gcf()

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Legger til en kantlinje (ramme) rundt hele figuren
    fig.patch.set_edgecolor('black')  # Farge på kantlinjen
    fig.patch.set_linewidth(2)        # Bredde på kantlinjen    
    plt.savefig(f"{output_file_prefix}_x.png")

    plt.show()
    plt.close()

    # Plotting the histogram of y distances
    plt.figure(figsize=(10, 6))
    plt.hist(y_values, bins=bins, alpha=0.75, color='red', label='y Distance (mip-PC)', range=range)
    plt.title(f'{title} y Component Distances (mip-PC)')
    plt.xlabel('y Distance')
    plt.ylabel('Frequency')
    plt.annotate(f'Mean (µ): {mean_y:.2f}\nStd Dev (σ): {std_y:.2f}', xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)
    plt.grid(True)
    
    
    
    plt.savefig(f"{output_file_prefix}_y.png")
    plt.show()

    plt.close()
    print(f"Plots saved to {output_file_prefix}_x.png and {output_file_prefix}_y.png")
import matplotlib.pyplot as plt

def plot_ckovRecon_by_pdg(df, pdg_codes, ckov_col='TrackAttributes_ckovReconThisTrack', pdg_col='mcTruth_pdgCodeTrack;'):
    """
    Plots histograms of reconstructed Cherenkov values split by specified PDG codes.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    pdg_codes (list of int): List of PDG codes to plot.
    ckov_col (str): Column name for Cherenkov values.
    pdg_col (str): Column name for PDG codes.
    """
    plt.figure(figsize=(12, 8))


    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))  # Set up a figure with two subplots side by side

    # Loop through the specified PDG codes and plot a histogram for each
    for pdg in pdg_codes:

        pdg_codes_of_interest = [pdg, 1111]
        # Filter and take absolute values
        filtered_series = df[pdg_col].abs()
        filtered_series = filtered_series[filtered_series.isin(pdg_codes_of_interest)]

        print(filtered_series)
        print(type(filtered_series))

        #filtered_series = pd.Categorical(filtered_series, categories=pdg_codes_of_interest, ordered=True)

        # Filter data for the current PDG code
        mask = df[pdg_col].abs() == pdg  # Use absolute value to ignore charge
        subset = df.loc[mask, ckov_col].dropna()  # Drop NaN values in the Cherenkov column

        pdg_code = df["mcTruth_pdgCodeTrack;"]


        ckov = df["TrackAttributes_ckovReconThisTrack"]
        ckov_mask = ckov >  0


        ckovmh = df["TrackAttributes_ckovReconMassHypThisTrack"]
        ckovmh_mask = ckovmh >  0

        pdg_code_of_interest = int(pdg)

        pdg_mask = df[pdg_col].abs() == pdg_code_of_interest
        combined_mask = pdg_mask & ckov_mask

        # Apply combined mask to DataFrame
        filtered_df = df.loc[combined_mask]
        pdg_mask = pdg_code == int(pdg)
        print(f"Values of TrackAttributes_ckovReconThisTrack for PDG {pdg_code_of_interest}:")
        print(filtered_df['TrackAttributes_ckovReconThisTrack'])


        print(f"filtered_df shape {pdg_mask.shape}")
        print(f"Number of entries with PDG {pdg_code_of_interest}: {pdg_mask.sum()}")  # .sum() on a boolean mask will count True values

        print(f"Number of entries wckov_mask  {ckov_mask.sum()}")  # .sum() on a boolean mask will count True values
        ckovpdg_mask = ckov_mask & pdg_mask
        print(f"Number of entries ckovpdg_mask  {ckovpdg_mask.sum()}")  # .sum() on a boolean mask will count True values

        print(type(pdg_code))

        print("First 10 PDG codes where Cherenkov mask is True:")
        print(filtered_df[pdg_col].head(10))
        
        print(f"ckov shape {ckov.shape}")
        print(f"ckov_mask {ckov[ckov_mask]}")
        print(f"pdg_mask {ckov[pdg_mask]}")
        print(f"ckov {ckov[ckov_mask & pdg_mask]}")
        print(f"For pdg {pdg} : mask {subset}")
        print(f"For pdg {pdg} : mask {ckov}")
        
        #ckov_mask = ckov >  0

        # Plot histogram
        plt.hist(ckov[ckov_mask], bins=50, alpha=0.75, range =(0.5, 0.8), label=f'PDG {pdg}')
        plt.title('Histogram of Reconstructed Cherenkov Values by PDG Code')
        plt.savefig("pdgCkov.png")

        plt.hist(ckovmh[ckovmh_mask], bins=50, alpha=0.75, range =(0.5, 0.8), label=f'PDG {pdg}')
        plt.title('Histogram of Reconstructed Cherenkov Values by mh')
        plt.savefig("mh.png")



        axs[0].hist(ckov[ckov_mask], bins=50, alpha=0.75, range =(0.5, 0.8), label=f'PDG {pdg} in {ckov_col}')
        axs[0].set_title(f'Histogram of {ckov_col} by PDG Code')
        axs[0].set_xlabel('Reconstructed Cherenkov Value')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].hist(ckovmh[ckovmh_mask], bins=50, alpha=0.75, range =(0.5, 0.8), label=f'PDG {pdg} in {ckov_col}')
        axs[1].set_title(f'Histogram of {ckov_col} by PDG Code')
        axs[1].set_xlabel('Reconstructed Cherenkov Value')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].grid(True)


    fig.savefig("pdgCkovSideBySide.png")  # Uncomment to save the figure to a file

    plt.title('Histogram of Reconstructed Cherenkov Values by PDG Code')
    plt.xlabel('Reconstructed Cherenkov Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig("pdgCkov.png")
    plt.close()


# Example usage:
# plot_ckovRecon_by_pdg(df=combined_df, pdg_codes=[211, 321, 2212])


def plot_momentum_per_specie(x_lims = None, all_dicts = None):
    """
    Plots momentum distribution per specie
    args:  
        x_lims : the limits of the x-axis, (momentum)
        all_dicts : dicts holding the data     
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    if x_lims is None :
        x_lims = (0, 5)

    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]

    # Filtering data based on absolute PDG codes
    pions = momentum[pdg == 211]
    kaons = momentum[pdg == 321]
    protons = momentum[pdg == 2212]

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot histograms and set x-axis limits
    axes[0].hist(pions.dropna(), bins='auto', color='blue', alpha=0.75)
    axes[0].set_title('Momentum Distribution for Pions')
    axes[0].set_xlabel('Momentum (GeV/c)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(x_lims[0], x_lims[1])  # Set x-axis limits

    axes[1].hist(kaons.dropna(), bins='auto', color='green', alpha=0.75)
    axes[1].set_title('Momentum Distribution for Kaons')
    axes[1].set_xlabel('Momentum (GeV/c)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(x_lims[0], x_lims[1])  # Set x-axis limits

    axes[2].hist(protons.dropna(), bins='auto', color='red', alpha=0.75)
    axes[2].set_title('Momentum Distribution for Protons')
    axes[2].set_xlabel('Momentum (GeV/c)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlim(x_lims[0], x_lims[1])  # Set x-axis limits

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def plot_numerical_histogram_nparr(data_array, output_file, **kwargs):
    # Pre-process the data if a range is specified.
    if 'range' in kwargs:
        range_min, range_max = kwargs['range']
        data_array = data_array[(data_array >= range_min) & (data_array <= range_max)]

    # Filter out NaN values from data
    clean_data = data_array[~np.isnan(data_array)]

    # Determine the number of bins
    num_bins = kwargs.pop('bins', 'auto')  # Use 'auto' as default if bins not specified in kwargs

    # Create the histogram
    n, bins, patches = plt.hist(clean_data, bins=num_bins, alpha=0.75, color='blue', **kwargs)

    # Calculate bin centers from bin edges
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Fit the Gaussian function to the histogram data
    popt, _ = curve_fit(gaussian, bin_centers, n, p0=[max(n), np.mean(clean_data), np.std(clean_data)])

    # Plot the Gaussian fit
    x_interval_for_fit = np.linspace(bins[0], bins[-1], 1000)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='Gaussian fit', color='red')

    # Annotate with the fit parameters
    plt.annotate(f'Mean (µ): {popt[1]:.4f}\nStd Dev (σ): {popt[2]:.4f}',
                 xy=(0.65, 0.55), xycoords='axes fraction', fontsize=12)

    # Set titles and labels
    plt.title(output_file)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()  # Show legend to indicate the Gaussian fit line
    plt.savefig(output_file)  # Save the figure to a file
    plt.show()  # Display the plot
    plt.close()  # Close the plot figure to free up memory
    print(f"Plot saved to {output_file}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def plot_numerical_histogram_nparr(data_array, output_file, **kwargs):

    gauss_fit = kwargs.pop('gauss_fit', False)


    # Pre-process the data if a range is specified.
    if 'range' in kwargs:
        range_min, range_max = kwargs.get('range')
        data_array = data_array[(data_array >= range_min) & (data_array <= range_max)]


    # Filter out NaN values from data
    clean_data = data_array[~np.isnan(data_array)]

    # Determine the number of bins
    num_bins = kwargs.pop('bins', 'auto')
    # Create the histogram
    n, bins, patches = plt.hist(clean_data, bins=num_bins, alpha=0.75, color='blue', **kwargs)

    # Calculate bin centers from bin edges
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Fit the Gaussian function to the histogram data

    if gauss_fit:
        try:

            popt, _ = curve_fit(gaussian, bin_centers, n, p0=[max(n), np.mean(clean_data), np.std(clean_data)])

            # Plot the Gaussian fit
            x_interval_for_fit = np.linspace(bins[0], bins[-1], 1000)
            plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='Gaussian fit', color='red')

            # Annotate with the fit parameters
            plt.annotate(f'µ : {popt[1]:.4f} [rad]\n σ : {popt[2]:.4f} [rad]',
                        xy=(0.65, 0.55), xycoords='axes fraction', fontsize=12)

        except RuntimeError as e:
            print(f"Gaussian fit failed: {e}")

    # Set titles and labels
    plt.title(output_file)
    plt.xlabel('Cherenkov Angle [rad]')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend() 
    plt.savefig(output_file)
    plt.show()  
    plt.close()
    print(f"Plot saved to {output_file}")

def plot_numerical_histogram_nparr2(data_array, output_file, **kwargs):
    """
    Plots a numerical histogram for a specified NumPy array and saves it as a file.

    Args:
    data_array (numpy.ndarray): Array containing the numerical data.
    output_file (str): File path where the plot will be saved.
    """
    if 'range' in kwargs:
        range_min, range_max = kwargs['range']
        data_array = data_array[(data_array >= range_min) & (data_array <= range_max)]


    plt.figure(figsize=(10, 6))
    plt.hist(data_array[~np.isnan(data_array)], alpha=0.75, color='blue', **kwargs)  # Handle NaN values

    mean = np.mean(data_array)
    std = np.std(data_array)
    plt.annotate(f'Mean (µ): {mean:.3f}\nStd Dev (σ): {std:.5f}', xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)

    plt.title(output_file)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_file)  # Save the figure to a file
    plt.show()  # Close the plot figure to free up memory
    plt.close()  # Close the plot figure to free up memory
    print(f"Plot saved to {output_file}")


def plot_numerical_histogram_nparr2(data_array, output_file, **kwargs):
    """
    Plots a numerical histogram for a specified NumPy array and saves it as a file.

    Args:
    data_array (numpy.ndarray): Array containing the numerical data.
    output_file (str): File path where the plot will be saved.
    """
    if 'range' in kwargs:
        range_min, range_max = kwargs['range']
        data_array = data_array[(data_array >= range_min) & (data_array <= range_max)]


    plt.figure(figsize=(10, 6))
    plt.hist(data_array[~np.isnan(data_array)], alpha=0.75, color='blue', **kwargs)  # Handle NaN values


    mean = np.mean(data_array)
    std = np.std(data_array)
    plt.annotate(f'Mean (µ): {mean:.3f}\nStd Dev (σ): {std:.5f}', xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)

    plt.title(output_file)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_file)  # Save the figure to a file
    plt.show()  # Close the plot figure to free up memory
    plt.close()  # Close the plot figure to free up memory
    print(f"Plot saved to {output_file}")




def plot_species_distributions_with_fits(all_dicts, m_l_cut, m_h_cut, z_score_threshold, massHyp = False):
    
    
    if massHyp:
        ckov_method = 'TrackAttributes_ckovReconMassHypThisTrack'
        title = "With Mass Hypothesis"
    else :
        ckov_method = 'TrackAttributes_ckovReconThisTrack'
        title = "Standard HTM"
    # Extract relevant data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]
    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)

    # Masks for momentum and positivity
    momentum_mask = (momentum > m_l_cut) & (momentum < m_h_cut)
    ckov_recon_mask = (ckov_recon > 0)

    species_codes = [211, 321, 2212]
    colors = ['blue', 'green', 'red']
    labels = ['Pion', 'Kaon', 'Proton']
    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}

    # Plot histograms for true species with predicted species
    for true_code in species_codes:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Mask for the true species
        true_mask = (pdg == true_code)


        # Filter true species by momentum and ckov positivity
        true_data = ckov_recon[true_mask & momentum_mask & ckov_recon_mask]
        
        # Histogram for the true species
        
        # Gaussian fit for the true species
        mu_true, std_true = norm.fit(true_data)
        xmin_true, xmax_true = plt.xlim()
        x_true = np.linspace(xmin_true, xmax_true, 100)
        p_true = norm.pdf(x_true, mu_true, std_true)
        #plt.plot(x_true, p_true, color='black', linewidth=2)

        # Add labels and titles
        plt.title(f"Cherenkov Angle Distribution for True {species_labels[true_code]} \n Momentum {m_l_cut}-{m_h_cut} GeV/c' \n {title}")
        plt.xlabel('Cherenkov Angle [rad]')
        plt.ylabel('Count')
        #plt.legend(loc='upper left')
        #plt.legend(bbox_to_anchor=(-0.15, 1), loc='upper left', bbox_transform=plt.gca().transAxes)

        plt.grid(True)
        
        # Plot histograms for predicted species within the true species
        
        sum_species = 0 
        for i_pred, pred_code in enumerate(species_codes):

            z_score_mask = (np.abs(differences[i_pred]) < z_score_threshold)

            pred_mask = (predicted_species == pred_code) & true_mask & momentum_mask & ckov_recon_mask & z_score_mask
            
            

            data_pred = ckov_recon[pred_mask]
            sum_species += np.count_nonzero(data_pred)
        
        for i_pred, pred_code in enumerate(species_codes):

            z_score_mask = (np.abs(differences[i_pred]) < z_score_threshold)

            pred_mask = (predicted_species == pred_code) & true_mask & momentum_mask & ckov_recon_mask & z_score_mask
            
            
            
            data_pred = ckov_recon[pred_mask]
            n_specie = np.count_nonzero(data_pred) 
            plt.hist(data_pred, bins=30, color=colors[i_pred], alpha=0.3, label=f'Predicted: {species_labels[pred_code]} : {n_specie}, {100*n_specie/sum_species}%')


            plt.legend(bbox_to_anchor=(0, 1), loc='upper left', bbox_transform=plt.gca().transAxes)

            if true_code == pred_code:


                mu, std = norm.fit(data_pred)
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                ax.plot(x, p, color=colors[i_pred], linewidth=2)




                # Annotate mean and standard deviation
                plt.text(0.05, 0.75, f'{species_labels[pred_code]}\nμ: {mu:.3f}\nσ: {std:.3f}', transform=plt.gca().transAxes, color=colors[i_pred], fontsize=9, verticalalignment='top')


        plt.show()
    





    text_y_start = 0.7


    # Concatenate all predicted species distributions and fit Gaussians
    plt.figure(figsize=(10, 6))
    for i_pred, pred_code in enumerate(species_codes):
        # Filter all predicted species by momentum and ckov positivity

        z_score_mask = (np.abs(differences[i_pred]) < z_score_threshold)

        all_pred_mask = (predicted_species == pred_code) & momentum_mask & ckov_recon_mask & z_score_mask
        all_pred_data = ckov_recon[all_pred_mask]


        #true_mask = (pdg == true_code)


       # pred_mask = (predicted_species == pred_code) & true_mask

        # Histogram for all predicted species
        count_all, bins_all, ignored_all = plt.hist(all_pred_data, bins=30, color=colors[i_pred], alpha=0.6, label=f'Predicted: {species_labels[pred_code]}')



        # Gaussian fit for all predicted species
        mu_all, std_all = norm.fit(all_pred_data)
        xmin_all, xmax_all = plt.xlim()
        x_all = np.linspace(xmin_all, xmax_all, 100)
        p_all = norm.pdf(x_all, mu_all, std_all)
        plt.plot(x_all, p_all, color=colors[i_pred], linewidth=2)

        plt.text(0.05, text_y_start, f'{species_labels[pred_code]}\nμ: {mu_all:.3f}\nσ: {std_all:.3f}', transform=plt.gca().transAxes, color=colors[i_pred], fontsize=9, verticalalignment='top')

        #plt.text(0.05, text_y_start, f'{pred_code}\nμ: {mu:.3f}\nσ: {std:.3f}', transform=plt.gca().transAxes, color=colors[i_pred], fontsize=9, verticalalignment='top')
        text_y_start -= 0.1  # Move the next annotation down

    plt.title(f'Combined Cherenkov Angle Distributions for All Predicted Species \n Momentum {m_l_cut}-{m_h_cut} GeV/c \n  {title} ')
    plt.xlabel('Cherenkov Angle [rad]')
    plt.ylabel('Count')        


    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', bbox_transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()



#### TRUE LABELS #### 
def specie_histograms_filtered(all_dicts, m_l_cut, m_h_cut, z_score_threshold, massHyp = False):
        
    if massHyp:
        ckov_method = 'TrackAttributes_ckovReconMassHypThisTrack'
        title = "With Mass Hypothesis"
    else :
        ckov_method = 'TrackAttributes_ckovReconThisTrack'
        title = "Standard HTM"


            
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"][ckov_method]
    differences, predicted_species = make_ckov_prediction_cut(all_dicts, ckov_method)

    # Create masks for momentum and ckov_recon positivity
    momentum_mask = (momentum > m_l_cut) & (momentum < m_h_cut)
    ckov_recon_mask = (ckov_recon > 0)

    # Create masks for spatial filtering
    xMip = all_dicts["ThisTrack;1"]["TrackAttributes_xMipThisTrack"]
    yMip = all_dicts["ThisTrack;1"]["TrackAttributes_yMipThisTrack"]
    xpc = all_dicts["ThisTrack;1"]["TrackAttributes_xPCThisTrack"]
    ypc = all_dicts["ThisTrack;1"]["TrackAttributes_yPCThisTrack"]
    dy = yMip - ypc
    dx = xMip - xpc
    dist = np.sqrt(dx**2 + dy**2)
    dist_mask = dist < 2

    plt.figure(figsize=(10, 6))
    text_y_start = 0.9  # Start text annotation from the top
    species_codes = [211, 321, 2212]
    species_labels = {211: 'Pion', 321: 'Kaon', 2212: 'Proton'}

    ##### For True Labels
    for i, (specie_code, color, label) in enumerate(zip([211, 321, 2212], ['blue', 'green', 'red'], ['Pion', 'Kaon', 'Proton'])):
        specie_mask = (pdg == specie_code)


        z_score_mask = (np.abs(differences[i, :]) < z_score_threshold)  # Apply z_score threshold for each species prediction

        # Combined mask including predictions, distances, and momentum
        combined_mask = specie_mask & ckov_recon_mask & dist_mask & momentum_mask & z_score_mask

        data = ckov_recon[combined_mask]
        count, bins, ignored = plt.hist(data, bins=100, alpha=0.5, color=color, label=label)

        # Gaussian fitting
        mu, std = norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        p = p * max(count) / max(p)  # Scale to histogram height
        plt.plot(x, p, color=color, linewidth=2)

        plt.text(0.05, text_y_start, f'{label} (Code: {specie_code})\nμ: {mu:.3f}\nσ: {std:.3f}', transform=plt.gca().transAxes, color=color, fontsize=9, verticalalignment='top')
        text_y_start -= 0.1  # Move the text down for the next species

    plt.title(f' {title} \n Filtered Cherenkov Recon Distribution for Momentum {m_l_cut}-{m_h_cut} GeV/c')
    plt.xlabel('Cherenkov Reconstruction Angle (Degrees)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.xlim(0.2, 0.75)  # Adjusted limits to focus on the range of interest
    plt.grid(True)
    plt.show()


    # # Plot setup for each true species
    # for i, species_code in enumerate(species_codes):
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    #     plt.subplots_adjust(wspace=0.3)  # Adjust horizontal space between subplots
    #     true_mask = (pdg == species_code)
    #     momentum_mask = (momentum >= m_l_cut) & (momentum <= m_h_cut)
    #     text_y_start = 0.9  # Top of the plot in axes fraction

    #     for i_pred, (pred_code, color, label) in enumerate(zip(species_codes, ['blue', 'green', 'red'], ['Pion', 'Kaon', 'Proton'])):
    #         ax = axes[i_pred]

    #         # Combine masks for predicted species and Z-score filtering
    #         pred_mask = (predicted_species == pred_code) & true_mask
    #         z_score_mask = (np.abs(differences[i_pred]) < z_score_threshold)
    #         combined_mask = pred_mask & z_score_mask & momentum_mask

    #         # Filtered data for the histogram
    #         data = ckov_recon[combined_mask]

    #         # Histogram and Gaussian fit
    #         count, bins, ignored = ax.hist(data, bins=30, color=color, alpha=0.6, label=f'{label} Predicted')
    #         mu, std = norm.fit(data)
    #         xmin, xmax = ax.get_xlim()
    #         x = np.linspace(xmin, xmax, 100)
    #         p = norm.pdf(x, mu, std)
    #         ax.plot(x, p, color='black', linewidth=2)




    #         # Annotate mean and standard deviation
    #         plt.text(0.05, text_y_start, f'{specie_code}\nμ: {mu:.3f}\nσ: {std:.3f}', transform=plt.gca().transAxes, color=color, fontsize=9, verticalalignment='top')
    #         text_y_start -= 0.1  # Move the next annotation down


    #         # Adding labels and titles
    #         ax.set_title(f'{species_labels[species_code]} (True), {label} (Predicted)')
    #         ax.set_xlabel('Cherenkov Angle [rad]')
    #         if i_pred == 0:
    #             ax.set_ylabel('Count')
    #         ax.legend()

    #     # Global title for the 3x1 figure
    #     fig.suptitle(f'Cherenkov Angle Distributions for True {species_labels[species_code]} \n {title}')

    # plt.show()
