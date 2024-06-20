from __future__ import print_function
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import normaltest, anderson
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dropout, Flatten
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dropout, Flatten

from tensorflow.keras.regularizers import l1, l2, l1_l2

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os


# specie_probability : vector containing the specie for same prob as specie in candidate_positions
def extract_neighborhood_map_new(x, y, mip_positions, attribute, attribute_key, neighborhood_size, map_size):
    """
    Extracts neighborhood maps based on candidate positions and MIP positions.


    Args:
        x (ndarray): X-coordinates of candidate positions.
        y (ndarray): Y-coordinates of candidate positions.
        mip_positions (ndarray): MIP positions.
        attribute (str): Attribute.
        attribute_key (str): Attribute key.
        neighborhood_size (int): Size of the neighborhood.
        map_size (int): Size of the map.

    Returns:
        ndarray: Neighborhood map around MIP, (num_samples x map_size x map_size)
                 Where Z-axis is the atrribute value.

    """

    cand_pos = np.asarray([x, y])

    print(f"mip_positions shape = {mip_positions.shape}")
    print(f"cand_pos shape = {cand_pos.shape}")

    num_samples = cand_pos.shape[0]
    num_candidates = cand_pos.shape[1]


    print(f"cand_pos shape = {cand_pos.shape}")
    #mip_positions = mip_positions.reshape((num_samples, 1, 2))

    # Use np.tile to replicate along the second dimension (num_candidates)
    mip_positions_expanded = np.tile(mip_positions, (1, num_candidates, 1))
    # Extend mip_positions to match the relevant dimensions of candidate_positions

    print(f"mip_positions_expanded shape = {mip_positions_expanded.shape}")
    print(f"cand_pos shape = {cand_pos.shape}")

    centered_positions = cand_pos - mip_positions_expanded

    # Now, extended_mip_positions has a shape of (8050, 1, 2, 4)

    # Perform the subtraction
    print(f"candidate_positions shape = {cand_pos.shape}")
    print(f"extended_mip_positions shape = {mip_positions_expanded.shape}")

    distances = cand_pos - mip_positions_expanded#[:, np.newaxis, :]

    print(f"candidate_positions shape = {cand_pos.shape}")

    # Calculate distances between candidate positions and MIP positions
    #distances = candidate_positions - mip_positions[:, np.newaxis, :]

    # Calculate the norm of distances to get the Euclidean distance
    distances = np.linalg.norm(distances, axis=-1)

    # Create an empty map
    neighborhood_maps = np.zeros((num_samples, map_size, map_size))

    # Check if the candidate falls within the neighborhood
    mask = distances <= neighborhood_size

    # Convert centered positions to map indices and shift them to be around the center of the map
    map_indices = np.round(centered_positions[:, :, :2] + map_size // 2).astype(int)
    map_indices = np.clip(map_indices, 0, map_size - 1)

    # Update the map
    # ef :change was done here: take the mask of specie_probability ; take only within +- map_size of MIP for candidate of specie
    neighborhood_maps[np.arange(num_samples)[:, np.newaxis], map_indices[:, :, 1], map_indices[:, :, 0]] = attribute[mask]

    # Plotting
    
    """
    for sample_idx in [1, 2]:
        plt.imshow(neighborhood_maps[sample_idx], cmap='gray')
        plt.colorbar()
        plt.title(f"Neighborhood Map for num_samples = {sample_idx} , attribute = {attribute_key}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Mark the MIP position (which should be at the center after centering)
        plt.scatter(map_size // 2, map_size // 2, c='red', marker='o')
        plt.show()
    """

    return neighborhood_maps


# specie_probability : vector containing the specie for same prob as specie in candidate_positions
# 
    """
    
def extract_neighborhood_map_new_vectorized(hadron_candidates, mip_positions, specie_probability, neighborhood_size, map_size):

    # get x, y
    candidate_positions = [hc.x, hc.y] for hc in hadron_candidates
    #candidate_positions = hc.prob for hc in hadron_candidates
  
    num_samples = candidate_positions.shape[0]


    print(f"mip_positions shape = {mip_positions.shape}")
    print(f"candidate_positions shape = {candidate_positions.shape}")


    num_samples = candidate_positions.shape[0]
    num_candidates = candidate_positions.shape[1]

    cand_pos = np.zeros((num_samples, num_candidates, 2))  # Create an array filled with zeros
    cand_pos = candidate_positions[:, :, :2]


    print(f"cand_pos shape = {cand_pos.shape}")
    mip_positions = mip_positions.reshape((num_samples, 1, 2))

    # Use np.tile to replicate along the second dimension (num_candidates)
    mip_positions_expanded = np.tile(mip_positions, (1, num_candidates, 1))
    # Use broadcasting to expand along the second dimension to get shape (8050, 915, 2)
    # Extend mip_positions to match the relevant dimensions of candidate_positions
    centered_positions = cand_pos - mip_positions_expanded


    # Now, extended_mip_positions has a shape of (8050, 1, 2, 4)

    # Perform the subtraction
    print(f"candidate_positions shape = {candidate_positions.shape}")
    print(f"extended_mip_positions shape = {mip_positions_expanded.shape}")

    distances = cand_pos - mip_positions_expanded#[:, np.newaxis, :]

    print(f"candidate_positions shape = {candidate_positions.shape}")

    # Calculate distances between candidate positions and MIP positions
    #distances = candidate_positions - mip_positions[:, np.newaxis, :]

    # Calculate the norm of distances to get the Euclidean distance
    distances = np.linalg.norm(distances, axis=-1)

    # Create an empty map
    neighborhood_maps = np.zeros((num_samples, map_size, map_size))

    # Check if the candidate falls within the neighborhood
    mask = distances <= neighborhood_size

    
  
    # Convert centered positions to map indices and shift them to be around the center of the map
    map_indices = np.round(centered_positions[:, :, :2] + map_size // 2).astype(int)
    map_indices = np.clip(map_indices, 0, map_size - 1)

    # Update the map
    # ef :change was done here: take the mask of specie_probability ; take only within +- map_size of MIP for candidate of specie
    neighborhood_maps[np.arange(num_samples)[:, np.newaxis], map_indices[:, :, 1], map_indices[:, :, 0]] = specie_probability[mask]

    # Plotting
    for sample_idx in [1, 2]:
        plt.imshow(neighborhood_maps[sample_idx], cmap='gray')
        plt.colorbar()
        plt.title(f"Neighborhood Map for num_samples = {sample_idx}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Mark the MIP position (which should be at the center after centering)
        plt.scatter(map_size // 2, map_size // 2, c='red', marker='o')

        plt.show()

    return neighborhood_maps

    """