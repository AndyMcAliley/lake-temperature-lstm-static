import os
import numpy as np
import pandas as pd

configfile: "2_process/process_config.yaml"


"""
This snakefile processes the data, preparing training, validation, and test
sets for neural network training. Two data sources are used to form two
distinct training sets: the MNTOHA data release and the
lake-temperature-model-prep pipeline. Some rules and functions apply to only
one data source, while some apply to both. If a rule or function has mntoha or
model_prep in its name, then you can be sure it only applies to that data
source. 

This snakefile is organized into four sections:

1. Interpolate observations to specified depths
2. Prepare lake metadata
3. Prepare fixed length sequences for the neural network
4. Summarize and group sequences into training, validation, and testing sets

Sections 1-3 are sub-divided by data source.
"""


#####################################################
#  1. Interpolate observations to specified depths  #
#####################################################


##### MNTOHA #####

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
    ).output.folder
    return(unzipped_archive)


def get_obs_file(file_category, archive_name, obs_file_name):
    """
    Return temperature observations filepath.

    Depend on unzip_archive checkpoint to ensure that
    the observation file gets unzipped.

    :param file_category: Category of files, e.g., obs_mntoha. Used by
        unzip_archive to determine parent directory.
    :param archive_name: Name of zip archive containing observation file, and
        name of directory to unzip to
    :param obs_file_name: name of unzipped observation file
    :returns: Path of temperature observations csv

    """
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


##### model-prep #####

# Convert 7b_temp_merge/out/merged_temp_data_daily.feather to csv
rule convert_model_prep_obs_to_csv:
    input:
        in_file="2_process/in/model_prep/temperature_observations/merged_temp_data_daily.feather"
    output:
        csv_file="2_process/tmp/model_prep/temperature_observations.csv"
    script:
        "2_process/src/convert_feather_to_csv.py"


# Add column of observation depths interpolated to nearest modeling mesh node
rule interpolate_model_prep_obs_depths:
    input:
        "2_process/tmp/model_prep/temperature_observations.csv"
    output:
        "2_process/tmp/model_prep/temperature_observations_interpolated.csv"
    params:
        depths=config["depths"]
    script:
        "2_process/src/make_obs_interpolated.py"


##############################
#  2. Prepare lake metadata  #
##############################


##### MNTOHA #####

# Add elevation to MNTOHA lake metadata
rule augment_mntoha_lake_metadata:
    input:
        lake_metadata="1_fetch/out/lake_metadata.csv",
        elevation_metadata="1_fetch/out/surface/lake_metadata.csv"
    output:
        augmented_metadata="2_process/tmp/mntoha/lake_metadata_augmented.csv"
    params:
        latitude_column_name="centroid_lat",
        longitude_column_name="centroid_lon",
        elevation_column_name="elevation_m"
    script:
        "2_process/src/augment_lake_metadata_w_elevation.py"


##### model-prep #####

# Convert 8_viz/inout/lakes_summary.feather to csv
rule convert_model_prep_metadata_to_csv:
    input:
        in_file="2_process/in/model_prep/metadata/lakes_summary.feather"
    output:
        csv_file="2_process/tmp/model_prep/lake_metadata.csv"
    script:
        "2_process/src/convert_feather_to_csv.py"


# Convert 1_crosswalk_fetch/out/ned_centroid_elevations.feather to csv
rule convert_model_prep_elevations_to_csv:
    input:
        in_file="2_process/in/model_prep/metadata/ned_centroid_elevations.feather"
    output:
        csv_file="2_process/tmp/model_prep/ned_centroid_elevations.csv"
    script:
        "2_process/src/convert_feather_to_csv.py"


# Convert 7_config_merge/out/canonical_lakes_area.rds to csv
rule convert_model_prep_area_to_csv:
    input:
        in_file="2_process/in/model_prep/metadata/canonical_lakes_area.rds"
    output:
        csv_file="2_process/tmp/model_prep/canonical_lakes_area.csv"
    script:
        "2_process/src/convert_rds_to_csv.R"


# Convert 7_config_merge/out/nml_Kw_values.rds to csv
rule convert_model_prep_clarity_to_csv:
    input:
        in_file = "2_process/in/model_prep/metadata/nml_Kw_values.rds"
    output:
        csv_file = "2_process/tmp/model_prep/nml_Kw_values.csv"
    script:
        "2_process/src/convert_rds_to_csv.R"


# Convert 7_config_merge/out/nml_lake_depth_values.rds to csv
rule convert_model_prep_depth_to_csv:
    input:
        in_file = "2_process/in/model_prep/metadata/nml_lake_depth_values.rds"
    output:
        csv_file = "2_process/tmp/model_prep/nml_lake_depth_values.csv"
    script:
        "2_process/src/convert_rds_to_csv.R"


# Add areas from 7_config_merge/out/canonical_lakes_areas.rds to model-prep lake metadata
rule augment_model_prep_lake_metadata_with_area:
    input:
        lake_metadata="2_process/tmp/model_prep/lake_metadata.csv",
        feature_metadata="2_process/tmp/model_prep/canonical_lakes_area.csv"
    output:
        augmented_metadata="2_process/tmp/model_prep/lake_metadata_area.csv"
    params:
        # Name of area column in input file
        feature_column_in="areas_m2",
        # Name to give area column in output file
        feature_column_out="area"
    script:
        "2_process/src/augment_lake_metadata_w_feature.py"


# Add depth from 7_config_merge/out/nml_lake_depth_values.rds to model-prep lake metadata
rule augment_model_prep_lake_metadata_with_depth:
    input:
        lake_metadata="2_process/tmp/model_prep/lake_metadata_area.csv",
        feature_metadata="2_process/tmp/model_prep/nml_lake_depth_values.csv"
    output:
        augmented_metadata="2_process/tmp/model_prep/lake_metadata_depth.csv"
    params:
        # Name of depth column in input file
        feature_column_in="lake_depth",
        # Name to give depth column in output file
        feature_column_out="lake_depth"
    script:
        "2_process/src/augment_lake_metadata_w_feature.py"


# Add clarity from 7_config_merge/out/nml_Kw_values.rds to model-prep lake metadata
rule augment_model_prep_lake_metadata_with_clarity:
    input:
        lake_metadata="2_process/tmp/model_prep/lake_metadata_area.csv",
        feature_metadata="2_process/tmp/model_prep/nml_Kw_values.csv"
    output:
        augmented_metadata="2_process/tmp/model_prep/lake_metadata_clarity.csv"
    params:
        # Name of clarity column in input file
        feature_column_in="Kw",
        # Name to give clarity column in output file
        feature_column_out="Kw"
    script:
        "2_process/src/augment_lake_metadata_w_feature.py"


# Add elevation to model_prep lake metadata
# This is a checkpoint because lake_metadata_augmented.csv is needed to
# determine the lake_sequence file names later.
checkpoint augment_model_prep_lake_metadata_with_elevation:
    input:
        lake_metadata="2_process/tmp/model_prep/lake_metadata_depth.csv",
        # elevation_metadata="1_fetch/out/surface/lake_metadata.csv"
        elevation_metadata="2_process/tmp/model_prep/ned_centroid_elevations.csv"
    output:
        augmented_metadata="2_process/tmp/model_prep/lake_metadata_augmented.csv"
    params:
        latitude_column_name="latitude",
        longitude_column_name="longitude",
        elevation_column_name="elevation"
    script:
        "2_process/src/augment_lake_metadata_w_elevation.py"


##############################################################
#  3. Prepare fixed length sequences for the neural network  #
##############################################################


##### MNTOHA #####

def dynamic_filenames_mntoha(site_id, file_category):
    """
    Return the files that contain dynamic data that are needed to construct
    sequences for a given MNTOHA lake.

    This function also triggers four checkpoints:
    1. fetch_mntoha_metadata to get lake_metadata.csv
    2. unzip_archive for this lake's drivers
    3. unzip_archive for this lake's clarity
    4. unzip_archive for this lake's ice flags

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
rule lake_sequences_mntoha:
    input:
        lake_metadata_file = "2_process/tmp/mntoha/lake_metadata_augmented.csv",
        observations_file = "2_process/tmp/mntoha/temperature_observations_interpolated.csv",
        dynamic_files = lambda wildcards: dynamic_filenames_mntoha(wildcards.site_id, file_category='dynamic_mntoha')
    output:
        site_sequences_file = "2_process/out/mntoha/sequences/sequences_{site_id}.npy"
    params:
        temp_col = 'temp',
        depth_col = 'interpolated_depth',
        date_col = 'date',
        area_col = 'area',
        lon_col = 'centroid_lon',
        lat_col = 'centroid_lat',
        elevation_col = 'elevation',
        lake_depth_col = 'lake_depth',
        config = config
    script:
        "2_process/src/lake_sequences.py"


##### model-prep #####

# Convert 7_config_merge/out/nml_meteo_fl_values.rds to csv
# This is a checkpoint because nml_meteo_fl_values.csv is needed to
# determine driver files, and the lake_sequence file names later.
checkpoint convert_model_prep_meteo_crosswalk_to_csv:
    input:
        in_file = "2_process/in/model_prep/metadata/nml_meteo_fl_values.rds"
    output:
        csv_file = "2_process/tmp/model_prep/nml_meteo_fl_values.csv"
    script:
        "2_process/src/convert_rds_to_csv.R"


def dynamic_filenames_model_prep(site_id):
    """
    Return the file that contains dynamic driver data that are needed to
    construct sequences for a given lake in the lake-temperature-model_prep
    footprint. Since only dynamic drivers are available for model-prep data,
    not time-varying clarity and ice flags, return a list with only one
    element.

    :param site_id: NHDHR lake ID
    :returns: List of dynamic filename (One element because there are no clarity or ice flags files)

    """
    # Location of driver data files
    meteo_directory = config["meteo_directory"]
    # Make this function dependent on the meteo crosswalk because the crosswalk
    # is used to determine driver filenames
    # meteo_crosswalk_file = "2_process/tmp/model_prep/nml_meteo_fl_values.csv"
    meteo_crosswalk_file = checkpoints.convert_model_prep_meteo_crosswalk_to_csv.get().output.csv_file
    meteo_crosswalk = pd.read_csv(meteo_crosswalk_file)
    meteo_matches = meteo_crosswalk.loc[meteo_crosswalk['site_id']==site_id]
    if len(meteo_matches) == 0:
        raise FileNotFoundError(f'No dynamic drivers found for {site_id}')
    lake = meteo_matches.iloc[0]
    drivers_filename = lake['meteo_fl']
    # dynamic filenames
    drivers_file = os.path.join(meteo_directory, drivers_filename)
    return [drivers_file]


# Create .npz of input/output sequences for one lake to use for training and testing
rule lake_sequences_model_prep:
    input:
        lake_metadata_file = "2_process/tmp/model_prep/lake_metadata_augmented.csv",
        observations_file = "2_process/tmp/model_prep/temperature_observations_interpolated.csv",
        dynamic_files = lambda wildcards: dynamic_filenames_model_prep(wildcards.site_id)
    output:
        site_sequences_file = "2_process/out/model_prep/sequences/sequences_{site_id}.npz"
    params:
        temp_col = 'temp',
        depth_col = 'interpolated_depth',
        date_col = 'date',
        area_col = 'area',
        lon_col = 'longitude',
        lat_col = 'latitude',
        elevation_col = 'elevation',
        lake_depth_col = 'lake_depth',
        config = config
    script:
        "2_process/src/lake_sequences.py"


##################################################################################
#  4. Summarize and group sequences into training, validation, and testing sets  #
##################################################################################


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
    if data_source == 'mntoha':
        # Make this function dependent on lake metadata
        # Needed because lake metadata is used to determine lake_sequence_files
        # lake_metadata_file = "1_fetch/out/lake_metadata.csv"
        lake_metadata_file = checkpoints.fetch_mntoha_metadata.get().output[0]
        lake_metadata = pd.read_csv(lake_metadata_file)
        site_ids = list(lake_metadata.site_id)
    elif data_source == 'model_prep':
        # Make this function dependent on lake metadata and the meteo crosswalk
        # because both are used to determine lake_sequence_files
        # lake_metadata_file = "2_process/tmp/model_prep/lake_metadata_augmented.csv"
        lake_metadata_file = (
            checkpoints.augment_model_prep_lake_metadata_with_elevation
            .get().output.augmented_metadata
        )
        # meteo_crosswalk_file = "2_process/tmp/model_prep/nml_meteo_fl_values.csv"
        meteo_crosswalk_file = (
            checkpoints.convert_model_prep_meteo_crosswalk_to_csv
            .get().output.csv_file
        )
        lake_metadata = pd.read_csv(lake_metadata_file)
        meteo_crosswalk = pd.read_csv(meteo_crosswalk_file)
        # Only use sites that are in both lake_metadata and meteo_crosswalk
        # Use Python's set intersection to get a list of such sites
        site_ids_metadata = lake_metadata.site_id
        site_ids_crosswalk = meteo_crosswalk.site_id
        site_ids = list(set(site_ids_metadata) & set(site_ids_crosswalk))
    else:
        raise ValueError(f'Data source {data_source} not recognized')
    # Fill in the two replacement fields in sequence_file_template with the
    # data source and the lake site ID, respectively.
    lake_sequence_files = [
        sequence_file_template.format(data_source, site_id) 
        for site_id in site_ids
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
        # The `lake_sequences` elements of the sequence files have shape (# sequences, sequence length, # depths + # features)
        num_sequences = np.load(sequences_file)['lake_sequences'].shape[0] 
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
            '2_process/out/{}/sequences/sequences_{}.npz',
            wildcards.data_source
        )
    output:
        "2_process/out/{data_source}/sequences/{data_source}_sequences_summary.csv"
    run:
        save_sequences_summary(input, output[0])


# Create training and test data
rule create_training_data:
    input:
        lambda wildcards: get_lake_sequence_files(
            '2_process/out/{}/sequences/sequences_{}.npz',
            wildcards.data_source
        ),
        sequences_summary_file = "2_process/out/{data_source}/sequences/{data_source}_sequences_summary.csv"
    output:
        train_file = "2_process/out/{data_source}/train.npz",
        valid_file = "2_process/out/{data_source}/valid.npz",
        test_file = "2_process/out/{data_source}/test.npz",
        split_summary_file = "2_process/out/{data_source}/split_summary.csv"
    params:
        process_config = config
    script:
        "2_process/src/training_data.py"

