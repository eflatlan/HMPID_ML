import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def specie_scatter_plots(all_dicts, combined_df):
    # Prepare the data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    ckov_recon = combined_df['TrackAttributes_ckovReconThisTrack']
    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"]).reshape(-1, 1)
    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"]).reshape(-1, 1)
    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"]).reshape(-1, 1)
    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"])
    ckov_recon_mh2 = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconMassHypThisTrack"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]

    ckov_recon_mask = (ckov_recon>0) & (ckov_recon<1)

    p = np.linspace(0, 5, 5000)
    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938

    angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    xMip = all_dicts["ThisTrack;1"]["TrackAttributes_xMipThisTrack"]
    yMip = all_dicts["ThisTrack;1"]["TrackAttributes_yMipThisTrack"]
    xpc = all_dicts["ThisTrack;1"]["TrackAttributes_xPCThisTrack"]
    ypc = all_dicts["ThisTrack;1"]["TrackAttributes_yPCThisTrack"]

    print(f"ckov_recon_mask shape {ckov_recon_mask.shape} ")
    print(f"ckov_recon shape {ckov_recon.shape} ")

    dx = xMip - xpc
    dy = yMip - ypc

    dist = np.sqrt(dx*dx + dy*dy)
    dist_mask = dist < 2

    print(f"dist shape {dist.shape} ")
    print(f"dist_mask shape {dist_mask.shape} ")

    th_pion = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"])
    th_kaon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"])
    th_proton = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.suptitle('Cherenkov Angle vs Momentum \n Standard HTM\n', fontsize=12)

    # Set up the scatter plot
    plt.figure(figsize=(10, 6))
    i = 0
    for specie, color, label in zip([211, 321, 2212], ['blue', 'green', 'red'], ['Pion', 'Kaon', 'Proton']):
        specie_mask = (pdg == specie)

        momentum_masked = momentum[specie_mask & dist_mask & ckov_recon_mask]
        ckov_masked =  ckov_recon[specie_mask & dist_mask & ckov_recon_mask]

        num_entries = np.count_nonzero(ckov_masked)

        plt.scatter(momentum_masked, ckov_masked, alpha=0.5, label=f"Actual specie = {label}", color=color)
        axes[i].scatter(momentum_masked, ckov_masked, alpha=0.5, label=f"Actual specie = {label}", color=color)
        if i == 0:
            axes[i].plot(p, angle_pion, label='Theoretical Pion Cherenkov Angle', color='cyan', linestyle='--')
        elif i == 1:
            axes[i].plot(p, angle_kaon, label='Theoretical Kaon Cherenkov Angle', color='green', linestyle='--')
        elif i == 2:
            axes[i].plot(p, angle_proton, label='Theoretical Proton Cherenkov Angle', color='purple', linestyle='-')
        axes[i].text(0.03, 0.97, f'Entries: {num_entries}', transform=axes[i].transAxes,
                     fontsize=9, color=color, verticalalignment='top')

        axes[i].set_xlim(0, 5)
        axes[i].set_ylim(0, 0.85)

        if i == 0:
            axes[i].set_ylabel('Cherenkov Angle [rad]')

        axes[i].set_xlabel('Momentum (GeV/c)')
        i += 1

    plt.plot(p, angle_pion, label='Theoretical Pion Cherenkov Angle', color='cyan', linestyle='--')
    plt.plot(p, angle_kaon, label='Theoretical Kaon Cherenkov Angle', color='green', linestyle='--')
    plt.plot(p, angle_proton, label='Theoretical Proton Cherenkov Angle', color='purple', linestyle='--')


    plt.xlim(0, 5)  # Example limits for momentum in GeV/c
    plt.ylim(0, .85)  # Example limits for Cherenkov angle in degrees


    plt.title('Cherenkov Angle vs Momentum')
    plt.xlabel('Momentum (GeV/c)')
    plt.ylabel('Cherenkov Angle [rad]')
    plt.legend()
    fig.suptitle('Cherenkov Angle vs Momentum \n Standard HTM\n', fontsize=16)

    plt.grid(True)
    plt.show()




def specie_scatter_plots_mh(all_dicts, combined_df):
    # Prepare the data
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    ckov_recon = combined_df['TrackAttributes_ckovReconThisTrack']
    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"]).reshape(-1, 1)
    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"]).reshape(-1, 1)
    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"]).reshape(-1, 1)

    ckov_recon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconMassHypThisTrack"])
    ckov_recon_mh2 = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconMassHypThisTrack"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]

    ckov_recon_mh2_mask = ckov_recon_mh2>0
    ckov_recon_mask = (ckov_recon>0) & (ckov_recon<1)

    p = np.linspace(0, 5, 5000)
    mass_pion = 0.1396
    mass_kaon = 0.497648
    mass_proton = 0.938

    angle_pion = np.arccos(np.sqrt(p**2 + mass_pion**2) / (p * 1.2904))
    angle_kaon = np.arccos(np.sqrt(p**2 + mass_kaon**2) / (p * 1.2904))
    angle_proton = np.arccos(np.sqrt(p**2 + mass_proton**2) / (p * 1.2904))

    xMip = all_dicts["ThisTrack;1"]["TrackAttributes_xMipThisTrack"]
    yMip = all_dicts["ThisTrack;1"]["TrackAttributes_yMipThisTrack"]
    xpc = all_dicts["ThisTrack;1"]["TrackAttributes_xPCThisTrack"]
    ypc = all_dicts["ThisTrack;1"]["TrackAttributes_yPCThisTrack"]

    print(f"ckov_recon_mask shape {ckov_recon_mask.shape} ")
    print(f"ckov_recon shape {ckov_recon.shape} ")

    dx = xMip - xpc
    dy = yMip - ypc

    dist = np.sqrt(dx*dx + dy*dy)
    dist_mask = dist < 2

    print(f"dist shape {dist.shape} ")
    print(f"dist_mask shape {dist_mask.shape} ")

    th_pion = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovThPionThisTrack"])
    th_kaon = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovThKaonThisTrack"])
    th_proton = np.asarray(all_dicts["ThisTrack;1"]["TrackAttributes_ckovThProtonThisTrack"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.suptitle('Cherenkov Angle vs Momentum \n With Mass Hypothesis\n', fontsize=12)

    # Set up the scatter plot
    plt.figure(figsize=(10, 6))
    i = 0
    for specie, color, label in zip([211, 321, 2212], ['blue', 'green', 'red'], ['Pion', 'Kaon', 'Proton']):
        specie_mask = (pdg == specie)
        momentum_masked = momentum[specie_mask & dist_mask & ckov_recon_mask]
        ckov_masked =  ckov_recon[specie_mask & dist_mask & ckov_recon_mask]

        num_entries = np.count_nonzero(ckov_masked)

        plt.scatter(momentum_masked, ckov_masked, alpha=0.5, label=f"Actual specie = {label}", color=color)
        axes[i].scatter(momentum_masked, ckov_masked, alpha=0.5, label=f"Actual specie = {label}", color=color)
        axes[i].text(0.03, 0.97, f'Entries: {num_entries}', transform=axes[i].transAxes,
                     fontsize=9, color=color, verticalalignment='top')

        if i == 0:
            axes[i].plot(p, angle_pion, label='Theoretical Pion Cherenkov Angle', color='cyan', linestyle='-')
        elif i == 1:
            axes[i].plot(p, angle_kaon, label='Theoretical Kaon Cherenkov Angle', color='lime', linestyle='-')
        elif i == 2:
            axes[i].plot(p, angle_proton, label='Theoretical Proton Cherenkov Angle', color='purple', linestyle='-')

        axes[i].set_xlim(0, 5)
        axes[i].set_ylim(0, 0.85)
        if i == 0:
            axes[i].set_ylabel('Cherenkov Angle [rad]')

        axes[i].set_xlabel('Momentum (GeV/c)')

        i += 1



    plt.plot(p, angle_pion, label='Theoretical Pion Cherenkov Angle', color='cyan', linestyle='-')
    plt.plot(p, angle_kaon, label='Theoretical Kaon Cherenkov Angle', color='lime', linestyle='-')
    plt.plot(p, angle_proton, label='Theoretical Proton Cherenkov Angle', color='purple', linestyle='-')


    plt.xlim(0, 5)  # Example limits for momentum in GeV/c
    plt.ylim(0, .85)  # Example limits for Cherenkov angle in degrees



    plt.title('Cherenkov Angle vs Momentum')
    plt.xlabel('Momentum (GeV/c)')
    plt.ylabel('Cherenkov Angle [rad]')
    plt.legend()
    plt.title(f'Cherenkov Angle vs Momentum \n With Mass Hypothesis')

    plt.grid(True)
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def specie_histograms_filtered(all_dicts, m_l_cut, m_h_cut):
    pdg = np.abs(all_dicts["McTruth;1"]["mcTruth_pdgCodeTrack;"])
    momentum = all_dicts["ThisTrack;1"]["TrackAttributes_momentumThisTrack"]
    ckov_recon = all_dicts["ThisTrack;1"]["TrackAttributes_ckovReconThisTrack"]

    # Create the mask for ckov_recon positivity and within the momentum range
    ckov_recon_mask = (ckov_recon > 0)
    xMip = all_dicts["ThisTrack;1"]["TrackAttributes_xMipThisTrack"]
    yMip = all_dicts["ThisTrack;1"]["TrackAttributes_yMipThisTrack"]
    xpc = all_dicts["ThisTrack;1"]["TrackAttributes_xPCThisTrack"]
    ypc = all_dicts["ThisTrack;1"]["TrackAttributes_yPCThisTrack"]

    momentum_mask = (momentum > m_l_cut) & (momentum < m_h_cut)
    dy = yMip - ypc
    dx = xMip - xpc

    dist = np.sqrt(dx**2 + dy**2)
    dist_mask = dist < 2
    # Set up the histogram plot
    plt.figure(figsize=(10, 6))
    text_y_start = 0.9  # Top of the plot in axes fraction

    # Plot histograms for each species with the combined mask and add Gaussian fit
    for specie_code, color, label in zip([211, 321, 2212], ['blue', 'green', 'red'], ['Pion', 'Kaon', 'Proton']):
        specie_mask = (pdg == specie_code)
        combined_mask = specie_mask & ckov_recon_mask & dist_mask & momentum_mask
        data = ckov_recon[combined_mask]

        # Plot histogram for the filtered data
        count, bins, ignored = plt.hist(data, bins=100, alpha=0.5, color=color, label=label)

        # Fit the Gaussian function to the histogram data
        mu, std = norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        p = p * max(count) / max(p)  # scale to match histogram height
        plt.plot(x, p, color=color, linewidth=2)



        # Annotate mean and standard deviation
        plt.text(0.05, text_y_start, f'{specie_code}\nμ: {mu:.3f}\nσ: {std:.3f}', transform=plt.gca().transAxes, color=color, fontsize=9, verticalalignment='top')
        text_y_start -= 0.1  # Move the next annotation down

    # Add labels and title
    plt.title(f'Filtered ckov_recon Distribution for Momentum {m_l_cut}-{m_h_cut} GeV/c')
    plt.xlabel('Cherenkov Reconstruction Angle (Degrees)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.xlim(0.5, 0.75)
    plt.grid(True)
    plt.show()


