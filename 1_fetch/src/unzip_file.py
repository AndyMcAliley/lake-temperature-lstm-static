import os
import zipfile
import datetime

def unzip_file(filename, source_dir, destination_dir):
    """
    Unzip file, save results to another directory.

    :param filename: File to unzip, with full paths
    :param source_dir: Directory to pull zipped file from.
        This part of the path is removed from the beginning of filename to determine where to save unzipped files.
    :param destination_dir: Directory to save unzipped files to.
        In each file path, source_dir is replaced with destination_dir to determine where to save unzipped files.
    :returns: list of paths to unzipped files

    """
    # check that the path to filename includes source_dir
    if os.path.samefile(source_dir, os.path.commonpath([filename, source_dir])):
        relfile = os.path.relpath(filename, source_dir)
        destination_subdir = os.path.join(destination_dir, os.path.splitext(relfile)[0])
        with zipfile.ZipFile(filename, 'r') as zf:
            unzipped_files = zf.namelist()
            zf.extractall(destination_subdir)
    else:
        raise FileNotFoundError(f'The path to {filename} does not include the directory {source_dir}.')
    return unzipped_files


if __name__ == '__main__':
    unzipped_files = unzip_file(snakemake.input[0],
                                snakemake.params['source_dir'],
                                snakemake.params['destination_dir'])

    supplementary_file = snakemake.output[0]
    with open(supplementary_file, 'w') as f:
        f.write(str(datetime.date.today()))


