import numpy as np
import pandas as pd

# Key: ClusterCandidates;3 (dict)
# Key: ClusterData_xValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_yValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_qValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_sizeValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_thetaCerValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_phiCerValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_sigmaRingValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_pionProbs (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_kaonProbs (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_protonProbs (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_protonProbsNorm (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_kaonProbsNorm (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_pionProbsNorm (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_sumProbabilityTrack (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_rawSizeValues (numpy array) - Shape: (29850, 749), Dtype: float32
# Key: ClusterData_numRawClustersValues (numpy array) - Shape: (29850, 749), Dtype: float32

# Key: McTruth;1 (dict)
# Key: mcTruth_isTrackToReconKnownPdg (pandas Series) - Length: 29850, Dtype: bool
# Key: mcTruth_isMipMatchedCorrectly (pandas Series) - Length: 29850, Dtype: bool
# Key: mcTruth_pdgCodeTrack; (pandas Series) - Length: 29850, Dtype: int64
# Key: mcTruth_pdgCodeClu (pandas Series) - Length: 29850, Dtype: int64
# Key: mcTruth_numCountedPhotonsPC (pandas Series) - Length: 29850, Dtype: int64


# Key: SumProballTracks;1 (dict)
# Key: sumProbabilityAllTracks (numpy array) - Shape: (29850, 749), Dtype: float32

# Key: ThisTrack;1 (dict)
# Key: TrackAttributes_xMipThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_yMipThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_xRadThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_yRadThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_xPCThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_yPCThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_thetaPThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_phiPThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_momentumThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_qMipThisTrack (pandas Series) - Length: 29850, Dtype: int64
# Key: TrackAttributes_sizeMipThisTrack (pandas Series) - Length: 29850, Dtype: int64
# Key: TrackAttributes_mipPcDistThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_ckovThPionThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_ckovThKaonThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_ckovThProtonThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_refIndexThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_ckovReconThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_ckovReconMassHypThisTrack (pandas Series) - Length: 29850, Dtype: float64
# Key: TrackAttributes_numCkovHough (pandas Series) - Length: 29850, Dtype: int64
# Key: TrackAttributes_numCkovHoughMH (pandas Series) - Length: 29850, Dtype: int64

# Key: OtherTracks;1 (dict)
# Key: TrackAttributes_xMipsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_yMipsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_xRadsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_yRadsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_xPCsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_yPCsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_thetaPsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_phiPsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_momentumsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_qMipsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_sizeMipsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_mipPcDistsOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_ckovThPionOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_ckovThKaonOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_ckovThProtonOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_refIndexesOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_ckovReconOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32
# Key: TrackAttributes_ckovReconMassHypOtherTracks (numpy array) - Shape: (29850, 14), Dtype: float32

# Key: HighChargeClusters;1 (dict)
# Key: highChargeClu_x (numpy array) - Shape: (29850, 339), Dtype: float32
# Key: highChargeClu_y (numpy array) - Shape: (29850, 339), Dtype: float32
# Key: highChargeClu_q (numpy array) - Shape: (29850, 339), Dtype: float32
# Key: highChargeClu_size (numpy array) - Shape: (29850, 339), Dtype: float32



def make_data():
    data = {
        'ClusterData': {
            'xValues': np.random.randn(29850, 749).astype(np.float32),
            'yValues': np.random.randn(29850, 749).astype(np.float32),
            'qValues': np.random.randn(29850, 749).astype(np.float32),
            'sizeValues': np.random.randn(29850, 749).astype(np.float32),
            'thetaCerValues': np.random.randn(29850, 749).astype(np.float32),
            'phiCerValues': np.random.randn(29850, 749).astype(np.float32),
            'sigmaRingValues': np.random.randn(29850, 749).astype(np.float32),
            'pionProbs': np.random.randn(29850, 749).astype(np.float32),
            'kaonProbs': np.random.randn(29850, 749).astype(np.float32),
            'protonProbs': np.random.randn(29850, 749).astype(np.float32),
            'protonProbsNorm': np.random.randn(29850, 749).astype(np.float32),
            'kaonProbsNorm': np.random.randn(29850, 749).astype(np.float32),
            'pionProbsNorm': np.random.randn(29850, 749).astype(np.float32),
            'sumProbabilityTrack': np.random.randn(29850, 749).astype(np.float32),
            'rawSizeValues': np.random.randn(29850, 749).astype(np.float32),
            'numRawClustersValues': np.random.randn(29850, 749).astype(np.float32)
        },
        'McTruth': {
            'isTrackToReconKnownPdg': pd.Series([True] * 29850, dtype=bool),
            'isMipMatchedCorrectly': pd.Series([False] * 29850, dtype=bool),
            'pdgCodeTrack': pd.Series(np.random.randint(100, 500, 29850), dtype=np.int64),
            'pdgCodeClu': pd.Series(np.random.randint(100, 500, 29850), dtype=np.int64),
            'numCountedPhotonsPC': pd.Series(np.random.randint(0, 100, 29850), dtype=np.int64)
        },
        'SumProballTracks': {
            'sumProbabilityAllTracks': np.random.randn(29850, 749).astype(np.float32)
        },
        'ThisTrack': {
            'TrackAttributes': pd.DataFrame({
                'xMipThisTrack': np.random.randn(29850),
                'yMipThisTrack': np.random.randn(29850),
                'xRadThisTrack': np.random.randn(29850),
                'yRadThisTrack': np.random.randn(29850),
                'xPCThisTrack': np.random.randn(29850),
                'yPCThisTrack': np.random.randn(29850),
                'thetaPThisTrack': np.random.randn(29850),
                'phiPThisTrack': np.random.randn(29850),
                'momentumThisTrack': np.random.randn(29850),
                'qMipThisTrack': np.random.randint(-5, 6, 29850),
                'sizeMipThisTrack': np.random.randint(0, 100, 29850),
                'mipPcDistThisTrack': np.random.randn(29850),
                'ckovThPionThisTrack': np.random.randn(29850),
                'ckovThKaonThisTrack': np.random.randn(29850),
                'ckovThProtonThisTrack': np.random.randn(29850),
                'refIndexThisTrack': np.random.randn(29850),
                'ckovReconThisTrack': np.random.randn(29850),
                'ckovReconMassHypThisTrack': np.random.randn(29850),
                'numCkovHough': np.random.randint(0, 50, 29850),
                'numCkovHoughMH': np.random.randint(0, 50, 29850)
            })
        },
        'OtherTracks': {
            'TrackAttributes': pd.DataFrame({
                'xMipsOtherTracks': np.random.randn(29850, 14),
                'yMipsOtherTracks': np.random.randn(29850, 14),
                'xRadsOtherTracks': np.random.randn(29850, 14),
                'yRadsOtherTracks': np.random.randn(29850, 14),
                'xPCsOtherTracks': np.random.randn(29850, 14),
                'yPCsOtherTracks': np.random.randn(29850, 14),
                'thetaPsOtherTracks': np.random.randn(29850, 14),
                'phiPsOtherTracks': np.random.randn(29850, 14),
                'momentumsOtherTracks': np.random.randn(29850, 14),
                'qMipsOtherTracks': np.random.randn(29850, 14),
                'sizeMipsOtherTracks': np.random.randn(29850, 14),
                'mipPcDistsOtherTracks': np.random.randn(29850, 14),
                'ckovThPionOtherTracks': np.random.randn(29850, 14),
                'ckovThKaonOtherTracks': np.random.randn(29850, 14),
                'ckovThProtonOtherTracks': np.random.randn(29850, 14),
                'refIndexesOtherTracks': np.random.randn(29850, 14),
                'ckovReconOtherTracks': np.random.randn(29850, 14),
                'ckovReconMassHypOtherTracks': np.random.randn(29850, 14)
            })
        },
        'HighChargeClusters': {
            'x': np.random.randn(29850, 339).astype(np.float32),
            'y': np.random.randn(29850, 339).astype(np.float32),
            'q': np.random.randn(29850, 339).astype(np.float32),
            'size': np.random.randn(29850, 339).astype(np.float32)
        }
    }

    return data
