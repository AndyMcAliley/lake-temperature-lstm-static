import numpy as np
import pandas as pd

def all_dates_depths(lake_obs_interpolated, 
                     depths,
                     temp_col,
                     depth_col,
                     date_col,
                     pad_before_days=0,
                     pad_after_days=0,
                     min_days=1):
    """
    Create a DataFrame with one column per depth and one row for every day from
    `pad_before_days` days before the first observation until `pad_after_days`
    after the last observation. The values in the DataFrame are observed
    temperatures. Missing observations are filled in with np.NaN. The DataFrame
    will include at least `min_days` rows. The start of the range of dates will
    shifted earlier as necessary to ensure that `min_days` rows are included.

    :param lake_obs_interpolated: DataFrame of observations for one lake, interpolated to the depths in the list `depths`.
    :param depths: Full list of depths at which temperatures will be provided by the trained model.
    :param temp_col: Name of DataFrame column with temperature values. (Default value = 'temp')
    :param depth_col: Name of DataFrame column with depth values to match up with `depths`. (Default value = 'interpolated_depth')
    :param date_col: Name of DataFrame column with observation dates. (Default value = 'date')
    :param pad_before_days: Number of days to pad before the first observation (Default value = 0)
    :param pad_after_days: Number of days to pad after the last observation (Default value = 0)
    :param min_days: Minimum number of days included in output DataFrame (Default value = 1)
    :returns: DataFrame of observations with NaN values for missing observations.

    """

    pad_before_days_td = np.timedelta64(pad_before_days, 'D')
    pad_after_days_td = np.timedelta64(pad_after_days, 'D')
    min_days_td = np.timedelta64(min_days, 'D')
    one_day = np.timedelta64(1, 'D')
    # start_date is pad_before_days before the date of the earliest observation
    start_date = lake_obs_interpolated.date.min() - pad_before_days_td
    # end_date is pad_after_days after the date of the latest observation
    end_date = lake_obs_interpolated.date.max() + pad_after_days_td
    # Ensure that at least min_days days are included
    # If not, pad to the left (shift the start date earlier)
    if end_date - start_date + one_day < min_days_td:
        start_date = end_date - min_days_td + one_day
    date_range = pd.date_range(start=start_date, end=end_date)

    # Now take sparse observations specific to a few depths and a few days, 
    # and create a full DataFrame of all depths and days, with NaN values for missing observations.

    # Take only the columns we need
    obs = lake_obs_interpolated[[date_col, depth_col, temp_col]]
    # Average duplicate measurements (same day, same nearest depth)
    # The resulting DataFrame has a MultiIndex of (date, depth)
    obs_midx = obs.groupby([date_col, depth_col]).mean()
    # Create a full MultiIndex of all possible combinations of dates and depths
    midx = pd.MultiIndex.from_product([date_range, depths], names=[date_col, depth_col])
    # Expand to the full set of date ranges and depths set by the MultiIndex 
    # Values are NaN for date-depth combinations that have no observations
    obs_full = obs_midx.reindex(midx, fill_value=np.NaN)

    # Reshape to have one column per depth
    obs_full = obs_full.unstack()
    # unstack() generates two column levels: temp and interpolated depths
    # Here we flatten to one level, naming columns 'temp_0.0', 'temp_0.5', etc.
    obs_full.columns = [f'{i}_{j}' for i,j in obs_full.columns.values]
    # Now obs_full has dimensions of len(date_range) by len(depths)
    return obs_full


def assemble_lake_data(site_id,
                       metadata_augmented_file,
                       obs_interpolated_file, 
                       drivers_file,
                       clarity_file,
                       ice_flags_file,
                       temp_col,
                       depth_col,
                       date_col,
                       config):
    """
    Assemble features and observed temperatures from one lake into equal-length
    sequences for training/testing. Features are meteorological drivers,
    clarity, ice flags, and static lake attributes.

    :param site_id: ID of lake to process
    :param metadata_augmented_file: Lake metadata file, augmented with elevation
    :param obs_interpolated_file: Temperature observations file, interpolated
        to LSTM depths
    :param drivers_file: Meteorological drivers csv for this lake
    :param clarity_file: Time-varying lake clarity csv for this lake
    :param ice_flags_file: Time-varying ice flags csv for this lake
    :param config: Snakemake config for 2_process
    :returns: Numpy array of sequences with shape (# sequences,
        sequence_length, # depths + # features)

    """
    # Settings for forming LSTM training sequences
    # Length of each sequence in days
    sequence_length = config['sequence_length']
    # Days between the start of one sequence and the start of the next
    sequence_offset = config['sequence_offset']
    # Number of days before 1st obs, e.g., 100 days
    spinup_time = config['spinup_time'] 
    # Discretized lake depths to use for LSTM
    depths = np.array(config['depths'])

    # All temperature observations interpolated to discretized depths
    obs_interpolated = pd.read_csv(obs_interpolated_file, parse_dates=['date'])
    # Observations for this lake
    lake_obs_interpolated = obs_interpolated[obs_interpolated.site_id==site_id].copy()
    # Only form sequences if there are any observations in this lake
    if len(lake_obs_interpolated) == 0:
        lake_sequences = np.empty((0))
    else:
        # Read metadata with elevation
        lake_metadata_augmented = pd.read_csv(metadata_augmented_file)
        # NOTE: If there are duplicate sites in lake_metadata_augmented,
        # only the first will be used.
        # If there are no matches, an IndexError will be raised.
        lake = lake_metadata_augmented[lake_metadata_augmented.site_id==site_id].iloc[0,:]

        # Create one large DataFrame of temperatures over a contiguous range of days and all
        # discretized depths, with NaN values wherever/whenever there are no
        # observed temperatures. The date range is padded to 
        #     a) provide spin-up time
        #     b) ensure that the date range is at least sequence_length long
        obs_full = all_dates_depths(lake_obs_interpolated,
                                    depths,
                                    temp_col,
                                    depth_col,
                                    date_col,
                                    pad_before_days=spinup_time,
                                    min_days=sequence_length)
        # obs_full now has dimensions of len(date range) by len(depths)

        # Next, add features (drivers, clarity, ice flags, and static
        # attributes) to obs_full. There are some NaN values in the clarity and
        # ice flag data. There are multiple ways to deal with NaN values in
        # inputs: interpolate to fill them, skip over them, or remove sequences
        # containing NaNs entirely. At this point, allow NaN values into the
        # sites. Later code will handle NaN values.

        # Join drivers by date
        drivers = pd.read_csv(drivers_file, parse_dates=['time'], index_col='time')
        obs_full = obs_full.join(drivers, how='left')

        # Join clarity by date
        clarity = pd.read_csv(clarity_file, parse_dates=['date'], index_col='date')
        obs_full = obs_full.join(clarity, how='left')

        # Join ice flags by date
        ice_flags = pd.read_csv(ice_flags_file, parse_dates=['date'], index_col='date')
        obs_full = obs_full.join(ice_flags, how='left')

        # Get attributes
        obs_full['area'] = lake['area']
        obs_full['lon'] = lake['centroid_lon']
        obs_full['lat'] = lake['centroid_lat']
        obs_full['elevation'] = lake['elevation']
        # Now obs_full is one long timeseries with depth-specific temperatures,
        # drivers, clarity, ice_flags, and attributes as columns

        # Split into sequences

        # Let's make some rules for the sequences to save to disk:
        # 1. Sequences are sequence_length days long
        # 2. Each next sequence starts sequence_offset days later than the previous
        # 3. Discard sequences that lack observed temperatures
        # 4. We know going in that the first and last sequences contain observed temperatures

        # Convert dataframe to numpy array
        # Use float32 to save on memory and train the neural network more efficiently.
        obs_full_array = obs_full.to_numpy(dtype=np.float32)
        # The shape of obs_full_array is (len(date range), # depths + # features)
        # The first `len(depths)` elements in the second dimension are temperature observations.
        # Features are drivers, clarity, ice flags, and static attributes, in that order.

        # Create a strided view into obs_full_array
        # This creates a numpy object with sequences of length sequence_length
        # and offsets of sequence_offset, without making an in-memory copy of the array
        all_sequences = np.lib.stride_tricks.sliding_window_view(
            obs_full_array, sequence_length, axis=0
        )[::sequence_offset, :, :]
        # The shape of all_sequences is (# sequences, # depths + # features, sequence_length).

        # Only write sequences to file if they contain temperature observations
        # The first len(depths) elements in the second dimension are temperature observations.
        sequences_have_no_obs = np.all(np.isnan(all_sequences[:, :len(depths), :]), axis=(1, 2))
        sequences_have_obs = np.logical_not(sequences_have_no_obs)
        # Sequences to save
        lake_sequences = all_sequences[sequences_have_obs, :, :].transpose((0, 2, 1))
        # After transpose, shape is (# sequences, sequence_length, # depths + # features)

        # The final temperature can be non-NaN, but the strided view's
        # sequences probably don't include the final day of the time period due
        # to truncation. So, add one more sequence that contains the final
        # sequence_length days, unless all_sequences happened to line up
        # perfectly such that the last sequence included that final day.
        # Recall, the shape of obs_full_array is 
        # (len(date range), # depths + # features)
        if (obs_full_array.shape[0] - sequence_length) % sequence_offset != 0:
            lake_sequences = np.concatenate((lake_sequences, np.array([obs_full_array[-sequence_length:, :]])), axis=0)

    return lake_sequences


def main(site_id,
         out_file,
         metadata_augmented_file,
         obs_interpolated_file,
         drivers_file,
         clarity_file,
         ice_flags_file,
         temp_col,
         depth_col,
         date_col,
         config):
    """
    Process raw data for one MNTOHA lake
    Save those data to a numpy file

    :param site_id: site_id of lake to process
    :param out_file: Filename of npy file to save
    :param metadata_augmented_file: lake metadata file, augmented with elevation
    :param obs_interpolated_file: temperature observations file, interpolated
        to LSTM depths
    :param drivers_file: Meteorological drivers csv for this lake
    :param clarity_file: Time-varying lake clarity csv
    :param ice_flags_file: Time-varying ice flags csv for this lake
    :param config: Snakemake config for 2_process

    """
    lake_sequences = assemble_lake_data(
        site_id,
        metadata_augmented_file,
        obs_interpolated_file,
        drivers_file,
        clarity_file,
        ice_flags_file,
        temp_col,
        depth_col,
        date_col,
        config
    )
    np.save(out_file, lake_sequences, allow_pickle=True)


if __name__ == '__main__':
    main(snakemake.wildcards['site_id'],
         snakemake.output[0], # npy file
         snakemake.input[0], # metadata_augmented_file
         snakemake.input[1], # obs_interpolated_file
         snakemake.input[2], # drivers_file
         snakemake.input[3], # clarity_file
         snakemake.input[4], # ice_flags_file
         snakemake.params['temp_col'],
         snakemake.params['depth_col'],
         snakemake.params['date_col'],
         snakemake.params['config'])

