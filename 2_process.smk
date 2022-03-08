import os
import numpy as np
import pandas as pd

configfile: "2_process/process_config.yaml"


def mntoha_lake_sequence_files(wildcards):
    """
    List all MNTOHA lake sequence .npy files for training and testing.

    :param wildcards: Snakemake wildcards (empty).
    :returns: List of lake training/testing sequence files.

    """
    # make this function dependent on fetch_mntoha_metadata
    # needed because lake_metadata.csv is used to determine lake_sequence_files
    lake_metadata_file = checkpoints.fetch_mntoha_metadata.get(**wildcards).output[0]
    lake_metadata = pd.read_csv(lake_metadata_file)
    lake_sequence_files = [os.path.join('2_process/out/mntoha_sequences', f'sequences_{site_id}.npy') for site_id in lake_metadata.site_id]
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
        num_sequences = np.load(sequences_file).shape[0] 
        sequence_counts.append(num_sequences)
    df_counts = pd.DataFrame(data={
        'sequences_file': lake_sequence_files_input,
        'num_sequences': sequence_counts
    })
    df_counts.to_csv(summary_file, index=False)


def dynamic_filenames(site_id, file_category = 'dynamic_mntoha'):
    """
    Return the files that contain dynamic data that are needed to construct
    sequences for a given lake.

    This function also triggers four checkpoints:
    1. fetch_mntoha_metadata to get lake_metadata.csv
    2. unzip_archive for this lake's drivers
    3. unzip_archive for this lake's clarity
    3. unzip_archive for this lake's ice flags

    :param site_id: NHDHR lake ID
    :param file_category: category of files, e.g., dynamic_mntoha. Used by
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
    unzip_archive_drivers = checkpoints.unzip_archive.get(
        file_category=file_category,
        directory_name=drivers_directory
    ).output[0]
    clarity_directory = f'clarity_{lake.group_id}'
    unzip_archive_clarity = checkpoints.unzip_archive.get(
        file_category=file_category,
        directory_name=clarity_directory
    ).output[0]
    ice_flags_directory = f'ice_flags_{lake.group_id}'
    unzip_archive_ice_flags = checkpoints.unzip_archive.get(
        file_category=file_category,
        directory_name=ice_flags_directory
    ).output[0]

    # dynamic filenames
    drivers_file = f'1_fetch/out/{file_category}/{drivers_directory}/{lake.meteo_filename}'
    clarity_file = f'1_fetch/out/{file_category}/{clarity_directory}/gam_{lake.site_id}_clarity.csv'
    ice_flags_file = f'1_fetch/out/{file_category}/{ice_flags_directory}/pb0_{lake.site_id}_ice_flags.csv'
    return [drivers_file, clarity_file, ice_flags_file]


# Summarize training sequences
rule process_mntoha:
    input:
        mntoha_lake_sequence_files
    output:
        "2_process/out/mntoha_sequences/mntoha_sequences_summary.csv"
    run:
        save_sequences_summary(input, output[0])


# Create .npy of input/output sequences for one lake to use for training and testing
rule mntoha_lake_sequences:
    input:
        "2_process/tmp/mntoha/lake_metadata_augmented.csv",
        "2_process/tmp/mntoha/temperature_observations_interpolated.csv",
        lambda wildcards: dynamic_filenames(wildcards.site_id, file_category='dynamic_mntoha')
    output:
        "2_process/out/mntoha_sequences/sequences_{site_id}.npy"
    params:
        config = config
    script:
        "2_process/src/lake_sequences_mntoha.py"


# Add elevation to MNTOHA lake metadata
rule augment_mntoha_lake_metadata:
    input:
        "1_fetch/out/lake_metadata.csv"
    output:
        "2_process/tmp/mntoha/lake_metadata_augmented.csv"
    script:
        "2_process/src/make_lake_metadata_augmented.py"


def mntoha_obs_file(wildcards):
    '''
    Return MNTOHA temperature observations filepath.

    Depend on unzip_archive checkpoint to ensure that
    temperature_observations.csv gets unzipped.

    :param wildcards: Snakemake wildcards (empty).
    :returns: Path of MNTOHA temperature observations csv

    '''
    # Trigger checkpoint to unzip temperature_observations.csv
    obs_file_directory = checkpoints.unzip_archive.get(
        file_category='obs_mntoha',
        directory_name='temperature_observations'
    ).output[0]
    return os.path.join('1_fetch', 'out', 'obs_mntoha', 'temperature_observations', "temperature_observations.csv")


# Add column of observation depths interpolated to nearest modeling mesh node
rule interpolate_mntoha_obs_depths:
    input:
        # "1_fetch/out/obs_mntoha/temperature_observations/temperature_observations.csv"
        mntoha_obs_file
    output:
        "2_process/tmp/mntoha/temperature_observations_interpolated.csv"
    params:
        depths=config["depths"]
    script:
        "2_process/src/make_obs_interpolated.py"

