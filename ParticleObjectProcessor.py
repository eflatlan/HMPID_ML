

import numpy as np
import pandas as pd

from map_helper_functions import extract_neighborhood_map_new


class ParticleObjectProcessorClass:
    def __init__(self, all_dicts, cluster_candidate_keys, this_track_keys, other_track_keys, high_charge_cluster_keys):
        

        self.this_track = None
        self.other_tracks = None
        self.cluster_candidates = None
        self.high_charge_clusters = None

        self.make_feature_vectors(all_dicts, this_track_keys, other_track_keys, high_charge_cluster_keys, cluster_candidate_keys)
        
        
        self.all_dicts = all_dicts
        self.make_vectors()
        
        self.maps = {}

    def make_feature_vectors(self, all_dicts, this_track_keys, other_track_keys, high_charge_cluster_keys, cluster_candidate_keys):
        
        filtered_clusters = self.filter_on_z_score(all_dicts, z_score_hadron_thresh=2)
        
        self.this_track = {key: all_dicts['ThisTrack'][key] for key in this_track_keys}
        #this_track_df = pd.DataFrame(this_track)

        self.other_tracks = {key: all_dicts['OtherTracks'][key] for key in other_track_keys}
        #other_tracks_df = pd.DataFrame(other_tracks)

        self.high_charge_clusters = {key: all_dicts['HighChargeClusters'][key] for key in high_charge_cluster_keys}
        #high_charge_clusters_df = pd.DataFrame(high_charge_clusters)
        
        self.cluster_candidates = {key: filtered_clusters['ClusterCandidates;3'][key] for key in cluster_candidate_keys}
        #cluster_candidates_df = pd.DataFrame(cluster_candidates)  
            


    def filter_on_z_score(self, all_dicts, z_score_hadron_thresh):
        # Key: TrackAttributes_ckovThPionThisTrack (pandas Series) - Length: 29850, Dtype: float64
        # Key: TrackAttributes_ckovThKaonThisTrack (pandas Series) - Length: 29850, Dtype: float64
        # Key: TrackAttributes_ckovThProtonThisTrack (pandas Series) - Length: 29850, Dtype: float64
        
        # Key: ClusterCandidates;3 (dict) : contains these fields
        # Key: ClusterData_sigmaRingValues (numpy array) - Shape: (29850, 749), Dtype: float32
        # Key: ClusterData_thetaCerValues (numpy array) - Shape: (29850, 749), Dtype: float32
        
        
        cluster_candidates = all_dicts['ClusterCandidates;3']
        
        sigma_ring =  cluster_candidates['ClusterData_sigmaRingValues']
        th_pion = all_dicts['ThisTrack;1']['TrackAttributes_ckovThPionThisTrack'] 
        th_kaon = all_dicts['ThisTrack;1']['TrackAttributes_ckovThKaonThisTrack'] 
        th_proton = all_dicts['ThisTrack;1']['TrackAttributes_ckovThProtonThisTrack'] 
        theta_cer = cluster_candidates['ClusterData_thetaCerValues']
        
        sum_prob_all_tracks =  all_dicts["SumProballTracks;1"]["sumProbabilityAllTracks"]       
        
        # ef : apply some maximum value for sigma_ring
        sigma_max_value = 0.0035
        sigma_ring = np.minimum(sigma_ring, sigma_max_value)

        print(f"shapes sigma_ring {sigma_ring.shape}, theta_cer {theta_cer.shape} th_pion {th_pion.shape}")

        print(f"types > th_pion : {type(th_pion)} sigma_ring : {type(sigma_ring)} theta_cer : {type(theta_cer)}" )

        th_pion_rs = th_pion.values.reshape(-1, 1)
        th_kaon_rs = th_kaon.values.reshape(-1, 1)
        th_proton_rs = th_proton.values.reshape(-1, 1)


        print(f"shapes sigma_ring {sigma_ring.shape}, theta_cer {theta_cer.shape} th_pion {th_pion.shape}")
        print(f"shapes reshaped > sigma_ring {sigma_ring.shape}, theta_cer {theta_cer.shape} th_pion_rs {th_pion_rs.shape}")

        z_pion = (th_pion_rs -theta_cer) / sigma_ring
        z_kaon = (th_kaon_rs - theta_cer) / sigma_ring
        z_proton = (th_proton_rs - theta_cer) /sigma_ring


        is_hadron_cand = (np.abs(z_pion) > z_score_hadron_thresh) | (np.abs(z_kaon) > z_score_hadron_thresh) | (np.abs(z_proton) > z_score_hadron_thresh)
        
        
        print("Shape of z_pion:", z_pion.shape)
        
        print("Shape of sigma_ring:", sigma_ring.shape)
        print("Shape of is_hadron_cand:", is_hadron_cand.shape)
        
        filtered_cluster_candidates = {
            'sigma_ring': cluster_candidates['ClusterData_sigmaRingValues'][is_hadron_cand],
            'theta_cer': cluster_candidates['ClusterData_thetaCerValues'][is_hadron_cand],
            'x': cluster_candidates['ClusterData_xValues'][is_hadron_cand],
            'y': cluster_candidates['ClusterData_yValues'][is_hadron_cand],
            'q': cluster_candidates['ClusterData_qValues'][is_hadron_cand],
            'size': cluster_candidates['ClusterData_sizeValues'][is_hadron_cand],
            'phi_cer': cluster_candidates['ClusterData_phiCerValues'][is_hadron_cand],
            'pion_probs': cluster_candidates['ClusterData_pionProbs'][is_hadron_cand],
            'kaon_probs': cluster_candidates['ClusterData_kaonProbs'][is_hadron_cand],
            'proton_probs': cluster_candidates['ClusterData_protonProbs'][is_hadron_cand],
            'pion_probs_norm': cluster_candidates['ClusterData_protonProbsNorm'][is_hadron_cand],
            'kaon_probs_norm': cluster_candidates['ClusterData_kaonProbsNorm'][is_hadron_cand],
            'proton_probs_norm': cluster_candidates['ClusterData_pionProbsNorm'][is_hadron_cand],
            'sum_prob_track': cluster_candidates['ClusterData_sumProbabilityTrack'][is_hadron_cand],
            'raw_cluster_size': cluster_candidates['ClusterData_rawSizeValues'][is_hadron_cand],
            'num_raw_clusters': cluster_candidates['ClusterData_numRawClustersValues'][is_hadron_cand],
            'z_pion': z_pion[is_hadron_cand],
            'z_kaon': z_kaon[is_hadron_cand],
            'z_proton': z_proton[is_hadron_cand],
            'sum_prob_all_tracks': sum_prob_all_tracks[is_hadron_cand],             
        }
        
        return filtered_cluster_candidates


    def make_maps(self, this_track, attributes, neighborhood_size, map_size):
        x_mips = this_track['TrackAttributes_xMipThisTrack']
        y_mips = this_track['TrackAttributes_yMipThisTrack']
        
        x = self.cluster_candidates['x']
        y = self.cluster_candidates['y']


        mip_positions = np.stack([x_mips, y_mips], axis=1)
        maps = {}
        for key in attributes:
            val = self.cluster_candidates[key]
            maps[key] = extract_neighborhood_map_new(x, y, mip_positions, val, key, neighborhood_size, map_size)
            
        
        high_charge_cluster_keys = ['highChargeClu_x','highChargeClu_y', 'highChargeClu_q', 'highChargeClu_size']
        x_high_charge = self.high_charge_clusters['highChargeClu_x']
        y_high_charge = self.high_charge_clusters['highChargeClu_x']
        q_high_charge = self.high_charge_clusters['highChargeClu_q']
        
        maps["high_charge"] = extract_neighborhood_map_new(x_high_charge, y_high_charge, mip_positions, q_high_charge, "High Charge Q", neighborhood_size, map_size)
        
        return maps

    def make_vectors(self):

        neighborhood_size = 100
        map_size = 100
        
        cnn_attributes = ['pion_probs', 'kaon_probs', 'proton_probs', 'pion_probs_norm', 'kaon_probs_norm', 'proton_probs_norm', 'sum_prob_track', 'raw_cluster_size', 'num_raw_clusters', 'z_pion', 'z_kaon', 'z_proton', 'sum_prob_all_tracks']
        cnn_attributes = ['z_pion', 'z_kaon', 'z_proton']
        self.make_maps(self.this_track, cnn_attributes, neighborhood_size, map_size)





cluster_candidate_keys = ['sigma_ring', 'theta_cer', 'x', 'y', 'q', 'size', 'phi_cer', 'pion_probs', 'kaon_probs', 'proton_probs', 'pion_probs_norm', 'kaon_probs_norm', 'proton_probs_norm', 'sum_prob_track', 'raw_cluster_size', 'num_raw_clusters', 'z_pion', 'z_kaon', 'z_proton', 'sum_prob_all_tracks'] 
this_track_keys = ['TrackAttributes_xMipThisTrack', 'TrackAttributes_yMipThisTrack', 'TrackAttributes_xRadThisTrack', 'TrackAttributes_yRadThisTrack', 'TrackAttributes_xPCThisTrack', 'TrackAttributes_yPCThisTrack', 'TrackAttributes_thetaPThisTrack', 'TrackAttributes_phiPThisTrack', 'TrackAttributes_momentumThisTrack', 'TrackAttributes_qMipThisTrack', 'TrackAttributes_sizeMipThisTrack', 'TrackAttributes_mipPcDistThisTrack', 'TrackAttributes_ckovThPionThisTrack', 'TrackAttributes_ckovThKaonThisTrack', 'TrackAttributes_ckovThProtonThisTrack', 'TrackAttributes_refIndexThisTrack', 'TrackAttributes_ckovReconThisTrack', 'TrackAttributes_ckovReconMassHypThisTrack', 'TrackAttributes_numCkovHough', 'TrackAttributes_numCkovHoughMH']
other_track_keys = ['TrackAttributes_xMipsOtherTracks', 'TrackAttributes_yMipsOtherTracks', 'TrackAttributes_xRadsOtherTracks', 'TrackAttributes_yRadsOtherTracks', 'TrackAttributes_xPCsOtherTracks', 'TrackAttributes_yPCsOtherTracks', 'TrackAttributes_thetaPsOtherTracks', 'TrackAttributes_phiPsOtherTracks', 'TrackAttributes_momentumsOtherTracks', 'TrackAttributes_qMipsOtherTracks', 'TrackAttributes_sizeMipsOtherTracks', 'TrackAttributes_mipPcDistsOtherTracks', 'TrackAttributes_ckovThPionOtherTracks', 'TrackAttributes_ckovThKaonOtherTracks', 'TrackAttributes_ckovThProtonOtherTracks', 'TrackAttributes_refIndexesOtherTracks', 'TrackAttributes_ckovReconOtherTracks', 'TrackAttributes_ckovReconMassHypOtherTracks']
high_charge_cluster_keys = ['highChargeClu_x','highChargeClu_y', 'highChargeClu_q', 'highChargeClu_size']


