import pandas as pd

from map_helper_functions import extract_neighborhood_map_new, extract_neighborhood_map_new_vectorized

plt.rcParams['figure.max_open_warning'] = 5000

class ParticleObject:
    def __init__(self):
        self.currentTrack = self.CurrentTrack()
        self.otherTrack = self.OtherTrack()
        
        self.highChargeClusters = self.HighChargeClusters()
        
        self.pionCandidates = self.HadronCandidates()
        self.kaonCandidates = self.HadronCandidates()
        self.protonCandidates = self.HadronCandidates()
        self.pion_map = None
        self.kaon_map = None
        self.proton_map = None

        self.pion_map_un = None
        self.kaon_map_un = None
        self.proton_map_un = None

        self.neighborhood_size = 50
        self.map_size = 50


    # extract_neighborhood_map_new
    # (hadron_candidates, mip_positions, specie_probability, neighborhood_size, map_size)
    def create_map(self):

        #print(f" in create_map fcn : ")
        mip_positions = [self.currentTrack.xMip, self.currentTrack.yMip]
        def __extract_map(hadron_candidates, attr = ""):

            # print(f" in __extract_map fcn : ")
            # print(f"type hadron_candidates : {type(hadron_candidates)}")

            x = hadron_candidates.x_padded# [cand.x_padded for cand in hadron_candidates]
            y = hadron_candidates.y_padded# [cand.x_padded for cand in hadron_candidates]

            prob = getattr(hadron_candidates, attr)

            #print(f" in __extract_map fcn : call extract_neighborhood_map_new")

            return extract_neighborhood_map_new(x, y, mip_positions, prob, self.neighborhood_size, self.neighborhood_size)

        #print(f"==========================\n\n\n\\n\n\n\n\\n\n\n====================================")

        # print(f"Number of nz per specie : {np.count_nonzero(self.pionCandidates.x_padded)}, {np.count_nonzero(self.kaonCandidates.x_padded)}, {np.count_nonzero(self.protonCandidates.x_padded)}")

        #print(f"Number of nz per specie : {self.pionCandidates.x_padded[self.pionCandidates.x_padded>0]}, {self.kaonCandidates.x_padded[self.kaonCandidates.x_padded>0]} {self.protonCandidates.x_padded[self.protonCandidates.x_padded>0]}")
        # print(f"Prob per specie : {self.pionCandidates.pion_prob_per_specie[self.pionCandidates.pion_prob_per_specie>0]}, {self.kaonCandidates.kaon_prob_per_specie[self.kaonCandidates.kaon_prob_per_specie>0]} {self.protonCandidates.proton_prob_per_specie[self.protonCandidates.proton_prob_per_specie>0]}")

        keys = ["x_padded", "y_padded", "theta_cer_padded", "sigma_ring_padded"]

        # for key in keys:
        #     print(f"Prob per  {key} : {getattr(self.pionCandidates, key)[self.pionCandidates.pion_prob_per_specie>0]}, {getattr(self.kaonCandidates, key)[self.kaonCandidates.kaon_prob_per_specie>0]}, {getattr(self.protonCandidates, key)[self.protonCandidates.proton_prob_per_specie>0]}")

        # class HadronCandidates:
        #     def __init__(self):
        #         self.x_padded = None
        #         self.y_padded = None
        #         self.q_padded = None
        #         self.size_padded = None
        #         self.phi_cer_padded = None
        #         self.theta_cer_padded = None
        #         self.sigma_ring_padded = None
        #         self.pion_prob_per_specie = None
        #         self.kaon_prob_per_specie = None
        #         self.proton_prob_per_specie = None
        #         self.L_track = None
        #         self.L_all_tracks = None

        #print(f"type self.pionCandidates : {type(self.pionCandidates)}")
        #         plt.imshow(neighborhood_maps[sample_idx], cmap='gray')
        #         plt.colorbar()
        #         plt.title(f"Neighborhood Map for num_samples = {sample_idx}")
        #         plt.xlabel("X-axis")
        #         plt.ylabel("Y-axis")

        #         # Mark the MIP position (which should be at the center after centering)
        #         plt.scatter(map_size // 2, map_size // 2, c='red', marker='o')
        #         plt.show()

        mip_x = self.currentTrack.xMip
        mip_y = self.currentTrack.yMip

        all_mip_x = self.otherTrack.xMip
        all_mip_y = self.otherTrack.yMip

        all_mip_x_filt = all_mip_x[all_mip_x > 0]
        all_mip_y_filt = all_mip_y[all_mip_y > 0]


        all_mip_ckov_theo = self.otherTrack.CkovTheoretical
        all_mip_ckov_theo_filt = all_mip_ckov_theo[all_mip_y > 0]

        verbose = False

        i = 0

        if verbose:
            print(f"n MIPs: ({np.count_nonzero((all_mip_x))})")

            for x in all_mip_x_filt:
                print(f"MIP : x{x} y{all_mip_y_filt[i]} | ThetaTheoretical = {all_mip_ckov_theo_filt}")
                i = i+1

            print(f"mip of current : x {mip_x} y {mip_y} | ckoc = {self.currentTrack.CkovTheoretical}")


        def extract_map_wrapper(hadron_candidates, attr):
            return __extract_map(hadron_candidates=hadron_candidates, attr=attr)



        from concurrent.futures import ThreadPoolExecutor, as_completed

        tasks = [
            (self.pionCandidates, "pion_prob_per_specie"),
            (self.kaonCandidates, "kaon_prob_per_specie"),
            (self.protonCandidates, "proton_prob_per_specie"),
            (self.pionCandidates, "pion_prob_per_specie_un"),
            (self.kaonCandidates, "kaon_prob_per_specie_un"),
            (self.protonCandidates, "proton_prob_per_specie_un")
        ]

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(extract_map_wrapper, hadron_candidates, attr): (hadron_candidates, attr) for hadron_candidates, attr in tasks}

            for future in as_completed(futures):
                hadron_candidates, attr = futures[future]
                try:
                    result_map, gbl_result = future.result()
                    if "pion" in attr:
                        if "un" in attr:
                            self.pion_map_un = result_map
                            gbl_pion_un = gbl_result
                        else:
                            self.pion_map = result_map
                            gbl_pion = gbl_result
                    elif "kaon" in attr:
                        if "un" in attr:
                            self.kaon_map_un = result_map
                            gbl_kaon_un = gbl_result
                        else:
                            self.kaon_map = result_map
                            gbl_kaon = gbl_result
                    elif "proton" in attr:
                        if "un" in attr:
                            self.proton_map_un = result_map
                            gbl_proton_un = gbl_result
                        else:
                            self.proton_map = result_map
                            gbl_proton = gbl_result
                except Exception as exc:
                    print(f"Generated an exception: {exc}")


        # self.pion_map, gbl_pion = __extract_map(hadron_candidates= self.pionCandidates,attr= "pion_prob_per_specie")
        # self.kaon_map, gbl_kaon = __extract_map(hadron_candidates=self.kaonCandidates, attr="kaon_prob_per_specie")
        # self.proton_map, gbl_proton = __extract_map(hadron_candidates=self.protonCandidates, attr="proton_prob_per_specie")


        # # ef : unnormalized (wrt sum_species = 1)
        # self.pion_map_un, gbl_pion_un = __extract_map(hadron_candidates= self.pionCandidates,attr= "pion_prob_per_specie_un")
        # self.kaon_map_un, gbl_kaon_un = __extract_map(hadron_candidates=self.kaonCandidates, attr="kaon_prob_per_specie_un")
        # self.proton_map_un, gbl_proton_un = __extract_map(hadron_candidates=self.protonCandidates, attr="proton_prob_per_specie_un")



        # if verbose:

        #     axes[0].imshow(self.pion_map[0, :, :], cmap='gray')
        #     axes[0].set_title("Pion Map")
        #     axes[0].text(mip_x, mip_y, f"MIP: ({mip_x}, {mip_y})", color='red')

        #     # Plot the kaon map
        #     axes[1].imshow(self.kaon_map[0, :, :], cmap='gray')
        #     axes[1].set_title("Kaon Map")

        #     # Plot the proton map
        #     axes[2].imshow(self.proton_map[0, :, :], cmap='gray')
        #     axes[2].set_title("Proton Map")


        #     fig1, axes1 = plt.subplots(1, 3, figsize=(8, 12))

        #     maps = [gbl_pion, gbl_kaon, gbl_proton]
        #     titles = ["Pion Map", "Kaon Map", "Proton Map"]

        #     for ax, map, title in zip(axes1, maps, titles):
        #         ax.imshow(map[0, :, :], cmap='gray')
        #         ax.scatter(all_mip_x[all_mip_x > 0], all_mip_y[all_mip_y > 0], color='green', marker='o')  # Plot all other MIPs as green dots
        #         ax.scatter(mip_x, mip_y, color='red', marker='o')  # Plot the MIP as a red dot
        #         ax.set_title(f"{title} Nonzero : {np.count_nonzero(map[0,:,:])}")


        #     # fig2, axes2 = plt.subplots(1, 3, figsize=(8, 12))
        #     # maps = [gbl_pion, gbl_kaon, gbl_proton]

        #     # for ax, map, title in zip(axes2, maps, titles):
        #     #     log_map = np.log1p(map[0, :, :])  # Apply logarithmic scaling
        #     #     ax.imshow(log_map, cmap='gray')
        #     #     ax.scatter(all_mip_x, all_mip_y, color='green', marker='o')  # Plot all other MIPs as green dots
        #     #     ax.scatter(mip_x, mip_y, color='red', marker='o')  # Plot the MIP as a red dot
        #     #     ax.set_title(title)
        #     plt.show()
        #     #map_size = nb_size = 50

        #     # X_map_pion = extract_neighborhood_map(candidate_positions = X_pion_candidates, mip_positions = X_mip_position, neighborhood_size = nb_size, map_size = map_size)


    def make_feature_vector(self, all_dicts, this_track_keys, other_track_keys, high_charge_cluster_keys, cluster_candidate_keys):

        cluster_candidate_keys = ['sigma_ring', 'theta_cer', 'x', 'y', 'q', 'size', 'phi_cer', 'pion_probs', 'kaon_probs', 'proton_probs', 'pion_probs_norm', 'kaon_probs_norm', 'proton_probs_norm', 'sum_prob_track', 'raw_cluster_size', 'num_raw_clusters', 'z_pion', 'z_kaon', 'z_proton', 'sum_prob_all_tracks'] 
        this_track_keys = ['TrackAttributes_xMipThisTrack', 'TrackAttributes_yMipThisTrack', 'TrackAttributes_xRadThisTrack', 'TrackAttributes_yRadThisTrack', 'TrackAttributes_xPCThisTrack', 'TrackAttributes_yPCThisTrack', 'TrackAttributes_thetaPThisTrack', 'TrackAttributes_phiPThisTrack', 'TrackAttributes_momentumThisTrack', 'TrackAttributes_qMipThisTrack', 'TrackAttributes_sizeMipThisTrack', 'TrackAttributes_mipPcDistThisTrack', 'TrackAttributes_ckovThPionThisTrack', 'TrackAttributes_ckovThKaonThisTrack', 'TrackAttributes_ckovThProtonThisTrack', 'TrackAttributes_refIndexThisTrack', 'TrackAttributes_ckovReconThisTrack', 'TrackAttributes_ckovReconMassHypThisTrack', 'TrackAttributes_numCkovHough', 'TrackAttributes_numCkovHoughMH']
        other_track_keys = ['TrackAttributes_xMipsOtherTracks', 'TrackAttributes_yMipsOtherTracks', 'TrackAttributes_xRadsOtherTracks', 'TrackAttributes_yRadsOtherTracks', 'TrackAttributes_xPCsOtherTracks', 'TrackAttributes_yPCsOtherTracks', 'TrackAttributes_thetaPsOtherTracks', 'TrackAttributes_phiPsOtherTracks', 'TrackAttributes_momentumsOtherTracks', 'TrackAttributes_qMipsOtherTracks', 'TrackAttributes_sizeMipsOtherTracks', 'TrackAttributes_mipPcDistsOtherTracks', 'TrackAttributes_ckovThPionOtherTracks', 'TrackAttributes_ckovThKaonOtherTracks', 'TrackAttributes_ckovThProtonOtherTracks', 'TrackAttributes_refIndexesOtherTracks', 'TrackAttributes_ckovReconOtherTracks', 'TrackAttributes_ckovReconMassHypOtherTracks']
        high_charge_cluster_keys = ['highChargeClu_x','highChargeClu_y', 'highChargeClu_q', 'highChargeClu_size']




    def make_feature_vector(self, all_dicts, this_track_keys, other_track_keys, high_charge_cluster_keys, cluster_candidate_keys):
        
        filtered_clusters = filter_on_z_score(all_dicts, z_score_hadron_thresh=2)
        
        this_track = {key: all_dicts['ThisTrack'][key] for key in this_track_keys}
        this_track_df = pd.DataFrame(this_track)

        other_tracks = {key: all_dicts['OtherTracks'][key] for key in this_track_keys}
        other_tracks_df = pd.DataFrame(other_tracks)

        high_charge_clusters = {key: all_dicts['HighChargeClusters'][key] for key in high_charge_cluster_keys}
        high_charge_clusters_df = pd.DataFrame(high_charge_clusters)
        
        cluster_candidates = {key: filtered_clusters['ClusterCandidates;3'][key] for key in cluster_candidate_keys}
        cluster_candidates_df = pd.DataFrame(cluster_candidates)  
        



    def __fill_track_attribute(self, key = "", pd = None):
        setattr(self.currentTrack, key, pd)

    def __fill_other_track(self, key = "", pd = None):
        setattr(self.otherTrack, key, pd)


    def fill_attribute(self, category, key, pd):


        if category == "PIONS":
            setattr(self.pionCandidates, key, pd)


        elif category == "KAONS":
            setattr(self.kaonCandidates, key, pd)
            #print(f"Filled field KAONS : key : {key} pd {pd}")

        elif category == "PROTONS":
            setattr(self.protonCandidates, key, pd)

        elif category == "CURRENT_TRACK":
            self.__fill_track_attribute(key, pd)
            #print(f"fille CURRENT_TRACK  : key {key}, pd {pd}")

        elif category == "OTHER_TRACK":
            self.__fill_other_track(key, pd)



    class HighChargeClusters:
        def __init__(self):
            self.x = None
            self.y = None
            self.charge = None
            self.size = None


    class CurrentTrack:
        def __init__(self):
            # Initialize fields for CURRENT_TRACK
            self.Momentum = None
            self.RefractiveIndex = None
            self.xRad = None
            self.yRad = None
            self.xMip = None
            self.yMip = None
            self.ThetaP = None
            self.PhiP = None
            self.CluCharge = None
            self.CluSize = None
            self.TrackPdg = None
            self.ckovReconstructed = None
            self.CkovTheoretical = None
            self.MIPS = None

    class OtherTrack:
        def __init__(self):
            # Initialize fields for OTHER_TRACK
            self.Momentum = None
            self.RefractiveIndex = None
            self.xRad = None
            self.yRad = None
            self.xMip = None
            self.yMip = None
            self.ThetaP = None
            self.PhiP = None
            self.CluCharge = None
            self.CluSize = None
            self.TrackPdg = None
            self.ckovReconstructed = None
            self.CkovTheoretical = None


    class HadronCandidates:
        def __init__(self):
            self.x_padded = None
            self.y_padded = None
            self.q_padded = None
            self.size_padded = None
            self.phi_cer_padded = None
            self.theta_cer_padded = None
            self.sigma_ring_padded = None
            self.pion_prob_per_specie = None
            self.kaon_prob_per_specie = None
            self.proton_prob_per_specie = None
            self.L_track = None
            self.L_all_tracks = None
            self.pion_prob_per_specie_un = None
            self.kaon_prob_per_specie_un = None
            self.proton_prob_per_specie_un = None



particle_objects = [ParticleObject() for _ in range(10000)]


