import os
import numpy as np
import pandas as pd

configfile: "2_process/process_config.yaml"


def trigger_unzip_archive(file_category, archive_name):
    """
    Trigger checkpoint to unzip zipped directory

    :param file_category: Category of files, e.g., dynamic_mntoha. Used by
        unzip_archive to determine parent directory.
    :param archive_name: Name of zip archive to be unzipped, and name of
        directory to unzip to

    :returns: Path to unzipped archive, relative to repository root directory
    """
    unzipped_archive = checkpoints.unzip_archive.get(
        file_category=file_category,
        archive_name=archive_name
    ).output[0]
    return(unzipped_archive)


def get_obs_file(file_category, archive_name, obs_file_name):
    '''
    Return temperature observations filepath.

    Depend on unzip_archive checkpoint to ensure that
    the observation file gets unzipped.

    :param file_category: Category of files, e.g., obs_mntoha. Used by
        unzip_archive to determine parent directory.
    :param archive_name: Name of zip archive containing observation file, and
        name of directory to unzip to
    :param obs_file_name: name of unzipped observation file
    :returns: Path of temperature observations csv

    '''
    # Trigger checkpoint to unzip observation file
    obs_file_directory = trigger_unzip_archive(file_category, archive_name)
    return os.path.join(obs_file_directory, obs_file_name)


# Add column of observation depths interpolated to nearest modeling mesh node
rule interpolate_mntoha_obs_depths:
    input:
        # "1_fetch/out/obs_mntoha/temperature_observations/temperature_observations.csv"
        lambda wildcards: get_obs_file(
            file_category='obs_mntoha',
            archive_name='temperature_observations',
            obs_file_name='temperature_observations.csv'
        )
    output:
        "2_process/tmp/mntoha/temperature_observations_interpolated.csv"
    params:
        depths=config["depths"]
    script:
        "2_process/src/make_obs_interpolated.py"


# Add elevation to MNTOHA lake metadata
rule augment_mntoha_lake_metadata:
    input:
        "1_fetch/out/lake_metadata.csv"
    output:
        "2_process/tmp/mntoha/lake_metadata_augmented.csv"
    script:
        "2_process/src/make_lake_metadata_augmented.py"


def dynamic_filenames(site_id, file_category):
    """
    Return the files that contain dynamic data that are needed to construct
    sequences for a given lake.

    This function also triggers four checkpoints:
    1. fetch_mntoha_metadata to get lake_metadata.csv
    2. unzip_archive for this lake's drivers
    3. unzip_archive for this lake's clarity
    3. unzip_archive for this lake's ice flags

    :param site_id: NHDHR lake ID
    :param file_category: Category of files, e.g., dynamic_mntoha. Used by
        unzip_archive to determine parent directory.
    :returns: List of 3 dynamic filenames: drivers, clarity, and ice flags

    """
    # make this function depend on fetch_mntoha_metadata
    # needed because lake_metadata.csv is used to determine dynamic files
    lake_metadata_file = checkpoints.fetch_mntoha_metadata.get().output[0]
    lake_metadata = pd.read_csv(lake_metadata_file)
    lake = lake_metadata.loc[lake_metadata['site_id']==site_id].iloc[0]
    # also make this function depend on unzip_archive
    # needed to link unzipped files with unzip_archive rule
    drivers_directory = f'inputs_{lake.group_id}'
    unzip_archive_drivers = trigger_unzip_archive(file_category, drivers_directory)
    clarity_directory = f'clarity_{lake.group_id}'
    unzip_archive_clarity = trigger_unzip_archive(file_category, clarity_directory)
    ice_flags_directory = f'ice_flags_{lake.group_id}'
    unzip_archive_ice_flags = trigger_unzip_archive(file_category, ice_flags_directory)

    # dynamic filenames
    drivers_file = f'{unzip_archive_drivers}/{lake.meteo_filename}'
    clarity_file = f'{unzip_archive_clarity}/gam_{lake.site_id}_clarity.csv'
    ice_flags_file = f'{unzip_archive_ice_flags}/pb0_{lake.site_id}_ice_flags.csv'
    return [drivers_file, clarity_file, ice_flags_file]


# Create .npy of input/output sequences for one lake to use for training and testing
rule mntoha_lake_sequences:
    input:
        "2_process/tmp/mntoha/lake_metadata_augmented.csv",
        "2_process/tmp/mntoha/temperature_observations_interpolated.csv",
        lambda wildcards: dynamic_filenames(wildcards.site_id, file_category='dynamic_mntoha')
    output:
        "2_process/out/mntoha_sequences/sequences_{site_id}.npy"
    params:
        temp_col = 'temp',
        depth_col = 'interpolated_depth',
        date_col = 'date',
        config = config
    script:
        "2_process/src/lake_sequences_mntoha.py"


def get_lake_sequence_files(sequence_file_template, data_source):
    """
    List all lake sequence .npy files for training and testing.

    :param sequence_file_template: Format string with two {} replacement
        fields. Serves as a Snakemake template for lake sequence .npy files.
        The first {} replacement field is for the data source, and the second
        {} replacement field is for the lake's site ID.
    :param data_source: Source of data, e.g., 'mntoha'. Used to read
        corresponding lake metadata file, and to construct list of lake
        sequence files.
    :returns: List of lake training/testing sequence files.

    """
    # Make this function dependent on lake metadata
    # Needed because lake metadata is used to determine lake_sequence_files
    if data_source == 'mntoha':
        lake_metadata_file = checkpoints.fetch_mntoha_metadata.get().output[0]
    else:
        raise ValueError(f'Data source {data_source} not recognized')
    lake_metadata = pd.read_csv(lake_metadata_file)
    # Fill in the two replacement fields in sequence_file_template with the
    # data source and the lake site ID, respectively.
    lake_sequence_files = [
        sequence_file_template.format(data_source, site_id) 
        for site_id in lake_metadata.site_id
    ]
    return lake_sequence_files


def save_sequences_summary(lake_sequence_files_input, summary_file):
    """
    Summarize the number of sequences with at least one temperature observation
    for each lake, and save the result to csv

    :param lake_sequence_files_input: the lake sequence files to summarize
    :param summary_file: csv file with how many sequences are in each lake

    """
    sequence_counts = []
    for sequences_file in lake_sequence_files_input:
        # Sequence files have shape (# sequences, sequence length, # depths + # features)
        num_sequences = np.load(sequences_file).shape[0] 
        sequence_counts.append(num_sequences)
    df_counts = pd.DataFrame(data={
        'sequences_file': lake_sequence_files_input,
        'num_sequences': sequence_counts
    })
    df_counts.to_csv(summary_file, index=False)


# Summarize training sequences
rule process_sequences:
    input:
        lambda wildcards: get_lake_sequence_files(
            '2_process/out/{}_sequences/sequences_{}.npy',
            wildcards.data_source
        )
    output:
        "2_process/out/{data_source}_sequences/{data_source}_sequences_summary.csv"
    run:
        save_sequences_summary(input, output[0])


# Create training and test data
rule create_training_data:
    input:
        sequences_summary_file = "2_process/out/{data_source}_sequences/{data_source}_sequences_summary.csv"
    output:
        train_file = "2_process/out/{data_source}/train.npz",
        test_file = "2_process/out/{data_source}/test.npz"
    params:
        train_frac = config['train_frac'],
        test_frac = config['test_frac'],
        n_depths = len(config['depths_all']),
        n_dynamic = len(config['dynamic_features_all']),
        n_static = len(config['static_features_all']),
        seed = config['seed']
    script:
        "2_process/src/training_data.py"

