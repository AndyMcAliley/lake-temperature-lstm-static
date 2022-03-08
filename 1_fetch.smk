import os


# Download lake_metadata.csv
# This is a checkpoint because lake_metadata.csv is needed to determine 
# later outputs; namely, the lake_sequence file names.
checkpoint fetch_mntoha_metadata:
    input:
        "1_fetch/in/pull_date.txt",
    output:
        "1_fetch/out/lake_metadata.csv"
    params:
        sb_id = "5e5c1c1ce4b01d50924f27e7"
    script:
        "1_fetch/src/sb_fetch.py"


# Download MNTOHA temperature observations zip file
rule fetch_mntoha_obs:
    input:
        "1_fetch/in/pull_date.txt"
    output:
        "1_fetch/tmp/obs_mntoha/temperature_observations.zip"
    params:
        sb_id = "5e5d0b68e4b01d50924f2b32"
    script:
        "1_fetch/src/sb_fetch.py"


# Download MNTOHA meteorological drivers, clarity, and ice flag zip files
rule fetch_mntoha_dynamic:
    input:
        "1_fetch/in/pull_date.txt"
    output:
        "1_fetch/tmp/dynamic_mntoha/{file}"
    params:
        sb_id = "5e5d0b96e4b01d50924f2b34"
    script:
        "1_fetch/src/sb_fetch.py"


# Unzip files from a zipped archive.
#
# This is a checkpoint because otherwise Snakemake won't track unzipped files
# from an archive and will delete or ignore them. The output is a directory
# because we don't know how many unzipped files there will be, but we know
# which directory they'll be in after they are unzipped.
checkpoint unzip_archive:
    input:
        "1_fetch/tmp/{file_category}/{directory_name}.zip",
        "1_fetch/in/pull_date.txt"
    output:
        "1_fetch/out/{file_category}/{directory_name}/date_of_unzipping.txt"
    params:
        source_dir = "1_fetch/tmp",
        destination_dir = "1_fetch/out",
        output_dir = "1_fetch/out/{file_category}/{directory_name}"
    script:
        "1_fetch/src/unzip_file.py"


