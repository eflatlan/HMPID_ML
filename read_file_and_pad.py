from scipy.stats import norm
import numpy as np
import pandas as pdf

import subprocess
import pkg_resources
import sys
required = {'uproot'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

from tensorflow.keras.preprocessing.sequence import pad_sequences

import subprocess
import sys
import uproot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

def pad_and_stack2(sequences, max_length=None):
	try:
		# Try padding, if max_length is not None, pad or truncate to that length
		padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', dtype='float32')  # Changed dtype to 'floar32'
		#print("padded ok")
	except ValueError:
		# Fallback: manually pad with zeros
		max_len = max_length if max_length is not None else max(len(seq) for seq in sequences)
		padded_sequences = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=0) for seq in sequences])
		#print("revert to other pad")

	return padded_sequences


def pad_and_stack3(sequences, max_length=None):
		# Check if sequences is a single numpy array and not a list of sequences.
		# If it's a single array, wrap it in a list.
		if isinstance(sequences, np.ndarray) and sequences.ndim == 1:
				sequences = [sequences]  # Wrap the single array in a list to create a list of sequences.

		try:
				# Try padding, if max_length is not None, pad or truncate to that length.
				padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', dtype='float32')  # dtype was corrected to 'float32'.
				#print("padded ok")
		except ValueError as e:
				print(f"pad_and_stack3 : ValueError: {e}")  # Print the error for debugging.
				# Fallback: manually pad with zeros if there's an error.
				max_len = max_length if max_length is not None else max(len(seq) for seq in sequences)
				padded_sequences = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=0) for seq in sequences])
				#print("reverted to manual padding")

		if padded_sequences.shape[1] == max_length:
				padded_sequences = padded_sequences.transpose()


		return padded_sequences



def test2_file(file_name="data.root"):
    all_dataframes = []  # List to hold all DataFrames
    all_dicts = {}  # List to hold all dicts

    with uproot.open(file_name) as file:
        for tree_name in file.keys():
            tree_name = tree_name.decode("utf-8") if isinstance(tree_name, bytes) else tree_name
            print(f"\nProcessing Tree: {tree_name}")
            tree = file[tree_name]
            df = tree_to_pandas(tree)

            print("Before padding:")
            print(df.head())
            print(f"shape {df.shape}")

            padded_df, event_data_dict = pad_dataframe(df)
            print("After padding:")
            print(padded_df.head())

            #print(f"shape {padded_df.shape}")

            for col in df.columns:
                nparr = np.asarray(df[col]).reshape(-1,1)
                print(f"name {col} shape {df[col].shape}")
                print(f"name {col} shape np {nparr.shape}")


            try:

                print("mcTruth_pdgCodeClu")
                print(df["mcTruth_pdgCodeClu"])

                print(df["mcTruth_pdgCodeClu"].dtype)

            except:
                print()

            all_dataframes.append(padded_df)

            all_dicts[tree_name] = event_data_dict

            try:
                print("Original data type:", df['mcTruth_pdgCodeTrack;'].dtype)
            except:
                print("wrong tree")


    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df, all_dicts


def pad_dataframe(df):
    df_out = pd.DataFrame()
    max_lengths = {}
    dtype_map = {}


    # First pass to determine max lengths and data types
    for col in df.columns:

        print(f"df[col] shape {df[col].shape}")
        df[col] = df[col].apply(lambda x: np.array(x, dtype=object) if isinstance(x, (list, np.ndarray)) else x)

        lengths = df[col].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
        max_lengths[col] = lengths.max()



        if max_lengths[col] > 0:

            first_element = df.loc[df[col].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0), col].iloc[0][0]
            dtype_map[col] = type(first_element)
        else:
            dtype_map[col] = df[col].dtype
            print(f"{col} : Target datatype from ROOT datatype {df[col].dtype}")



    col = df.columns[0]
    max_length = max_lengths[col]
    if max_length == 0:
        npz = np.zeros((len(df[col]), 1))
    else :
        npz = np.zeros((len(df[col]), max_length))

    #print(f"npz {npz.shape}")
    np_arrays = {col: np.empty((len(df), max_lengths[col]), dtype=dtype_map[col]) for col in df.columns}

    #df_out = pd.DataFrame({col: [np.zeros_like(npz) for _ in range(len(df))] for col in df.columns})


    dtype_list = [(col, dtype_map[col], (max_lengths[col],)) for col in df.columns]

    event_data_dict = {}  # add CkovHyps here?

    #print(f"max_length {max_length}")
    # Second pass to pad and enforce data type consistency
    for col, max_length in max_lengths.items():
        print(f"col {col}")
        event_data_dict[col] = np.zeros_like(npz)
        data_type = df[col].dtype
        #print(f"max_length {max_length}")

        # q_padded2 = pad_and_stack2(df[col], max_length=max_length)
        # print(f"q_padded2 {q_padded2.shape}")



        # df2 = pad_and_stack2(df[col], max_length=max_length)
        # print(f"df2 {df2.shape}")


        if max_length > 0:

            event_data_dict[col] =  pad_and_stack2(df[col], max_length=max_length)
            #print(f"col {col} event_data_dict[col] { event_data_dict[col].shape}")


            # sequences = df[col].tolist()
            # # Use the custom padding function to pad the sequences
            # padded_sequences = pad_and_stack2(sequences, max_length=max_length)

            # print(f"padded_sequences shape{padded_sequences.shape}")

            # # Assign the padded sequences back to the DataFrame
            # df[col] = padded_sequences#list(padded_sequences)
            # print(f"df[col] {df[col].shape}")

            # npz = np.zeros((len(df[col]), max_length))
            # target_dtype = dtype_map[col]  # Get the mapped data type
            # print(f"df[col] {df[col].shape}")

            # df_out[col] = padded_sequences
            # print(f"df_out[col] {df_out[col].shape}")

            # npz = df[col]

            # print(f"{col} : Target datatype {target_dtype}, from ROOT datatype {data_type}")
            # df[col] = df[col].apply(lambda x: np.pad(x.astype(target_dtype), (0, max_length - len(x)), 'constant') if isinstance(x, np.ndarray) and len(x) > 0 else np.zeros(max_length, dtype=target_dtype))

            # dfc = np.asarray(df[col])
            # print(dfc.shape)
            # print(type(dfc))
            # q_padded2 = pad_and_stack2(df[col], max_length=max_length)


            # print(f"q_padded2 {q_padded2.shape}")

            # print(df[col])
            # print(dfc.shape)

        else :
            #npz = np.zeros((len(df[col]), max_length))
            #df[col] = df[col].astype(dtype_map[col])
            event_data_dict[col] =  (df[col])
            #print(f"df[col] {event_data_dict[col].shape}")


    #print(f"event_data_dict length: {len(event_data_dict)}")

    for key, value in event_data_dict.items():
        print(f"{key}: {value.shape}")
    #print(f"event_data_dict length: {len(event_data_dict)}")


    #print(f"Num keys in event_data_dict {event_data_dict.items()}")

    return df, event_data_dict

def concatenate_trees(directory):
    tree_dict = {}  # Dictionary to store DataFrames for each tree type across all files

    # List all ROOT files in the specified directory
    filenames = glob.glob(f'{directory}/*.root')

    # Collect DataFrames for each tree from each file
    for filename in filenames:
        with uproot.open(filename) as file:
            for tree_name in file.keys():
                clean_tree_name = tree_name.decode('utf-8').rstrip(';1')
                if clean_tree_name not in tree_dict:
                    tree_dict[clean_tree_name] = []
                tree = file[clean_tree_name]
                df = tree_to_pandas(tree)
                tree_dict[clean_tree_name].append(df)

    all_dataframes = {}
    all_dicts = {}

    # Concatenate DataFrames for each tree type and process them
    for tree_name, dfs in tree_dict.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nProcessing combined tree: {tree_name}")
        print("Before padding:")
        print(combined_df.head())
        print(f"Shape {combined_df.shape}")

        padded_df, event_data_dict = pad_dataframe(combined_df)
        print("After padding:")
        print(padded_df.head())

        all_dataframes[tree_name] = padded_df
        all_dicts[tree_name] = event_data_dict

    return all_dataframes, all_dicts




def test2(directory):
    all_dataframes = {}  # Dictionary to hold combined DataFrames for each tree type
    all_dicts = {}       # Dictionary to hold additional data like event_data_dict for each tree type

    filenames = glob.glob(f'{directory}/*.root')  # Gets all .root files in the directory
    trees_collected = {}  # This will hold lists of DataFrames to be concatenated

    # First, collect all DataFrames corresponding to each tree across all files
    for filename in filenames:
        with uproot.open(filename) as file:
            for tree_name in file.keys():
                tree_name_cleaned = tree_name.decode("utf-8").strip(';1') if isinstance(tree_name, bytes) else tree_name
                if tree_name_cleaned not in trees_collected:
                    trees_collected[tree_name_cleaned] = []
                tree = file[tree_name_cleaned]
                df = tree_to_pandas(tree)
                trees_collected[tree_name_cleaned].append(df)

    # Now, concatenate DataFrames for each tree type and process them
    for tree_name, dfs in trees_collected.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Processing combined tree: {tree_name}")
        print("Before padding:")
        print(combined_df.head())
        print(f"shape {combined_df.shape}")

        padded_df, event_data_dict = pad_dataframe(combined_df)
        print("After padding:")
        print(padded_df.head())

        all_dataframes[tree_name] = padded_df
        all_dicts[tree_name] = event_data_dict

    return all_dataframes, all_dicts


def process_file(file_name):
    with uproot.open(file_name) as file:
        tree_versions = {}
        # Step A: Identify the highest version of each tree
        for tree_name in file.keys():
            # Check if tree_name is a bytes object and decode if necessary
            if isinstance(tree_name, bytes):
                tree_name = tree_name.decode('utf-8')
            base_name, version = tree_name.split(';')[0], int(tree_name.split(';')[1])
            if base_name not in tree_versions or version > tree_versions[base_name][1]:
                tree_versions[base_name] = (tree_name, version)

        # Step B: Process only the highest version of each tree
        all_dataframes = []
        all_dicts = {}
        for base_name, (tree_name, version) in tree_versions.items():
            tree = file[tree_name]
            df = tree_to_pandas(tree)
            padded_df, event_data_dict = pad_dataframe(df)
            all_dataframes.append(padded_df)
            all_dicts[tree_name] = event_data_dict
            print(f"Processed {tree_name} with version {version}")

        # Step C: Combine all processed DataFrames (optional)
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df, all_dicts


def test22(directory):
    all_dataframes = []  # List to hold all DataFrames
    all_dicts = {}  # Dictionary to hold all dicts

    filenames = glob.glob(f'{directory}/*.root')  # Gets all .root files in the directory
    for filename in filenames:
        with uproot.open(filename) as file:
            for tree_name in file.keys():
                tree_name = tree_name.decode("utf-8").strip(';1') if isinstance(tree_name, bytes) else tree_name
                print(f"\nProcessing Tree: {tree_name}")
                tree = file[tree_name]
                df = tree_to_pandas(tree)

                #print("Before padding:")
                #print(df.head())
                #print(f"shape {df.shape}")

                padded_df, event_data_dict = pad_dataframe(df)
                #print("After padding:")
                print(padded_df.head())

                for col in df.columns:
                    nparr = np.asarray(df[col]).reshape(-1,1)
                    #print(f"name {col} shape {df[col].shape}")
                    #print(f"name {col} shape np {nparr.shape}")

                # try:
                #     print("mcTruth_pdgCodeClu")
                #     print(df["mcTruth_pdgCodeClu"])
                #     print(df["mcTruth_pdgCodeClu"].dtype)
                # except KeyError:
                #     print("KeyError in accessing mcTruth_pdgCodeClu")

                all_dataframes.append(padded_df)
                all_dicts[tree_name] = event_data_dict

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df, all_dicts

def print_column_shapes(df):
    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Count the non-null entries in each column
        non_null_count = df[column].notna().sum()
        print(f"Column: {column}, Non-Null Count: {non_null_count} / {df[column].shape}")


def tree_to_pandas(tree):
    df = tree.arrays(library="pd")
    print("DataFrame shape:", df.shape)  # This line prints the shape of the DataFrame
    print(f"Number of entries in the tree: {tree.num_entries}")
    return df







import numpy as np
import pandas as pd

def print_dict_structure(data):
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"Key: {key} (dict)")
            print_dict_structure(value)  # Recursively print dictionary content
        elif isinstance(value, np.ndarray):
            # Print shape and data type (dtype) of the numpy array
            print(f"Key: {key} (numpy array) - Shape: {value.shape}, Dtype: {value.dtype}")
        elif isinstance(value, pd.Series):
            # Print length and data type (dtype) of the pandas Series
            print(f"Key: {key} (pandas Series) - Length: {len(value)}, Dtype: {value.dtype}")
        else:
            # Using type(value).__name__ to get the type as a string
            datatype = type(value).__name__
            print(f"Key: {key} - Type: {datatype}")