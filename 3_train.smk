import yaml

def train_config_path(data_source, run_id):
    """
    Get the path to a training config file, given the run_id and the model_id
    """
    return f"3_train/in/{data_source}/{run_id}.yaml"


# Save config files to output folder
rule save_config:
    input:
        process_config = "2_process/process_config.yaml",
        train_config = lambda wildcards: train_config_path(
            data_source=wildcards.data_source,
            run_id=wildcards.run_id
        )
    output:
        process_config = "3_train/out/{data_source}/{run_id}/{run_id}_process.yaml",
        train_config = "3_train/out/{data_source}/{run_id}/{run_id}_train.yaml"
    shell:
        """
        mkdir -p 3_train/out/{wildcards.data_source}/{wildcards.run_id}
        cp {input.process_config} {output.process_config}
        cp {input.train_config} {output.train_config}
        """


def read_train_config(data_source, run_id):
    """
    Read a training configuration yaml file and return a dictionary
    """
    config_path = train_config_path(data_source, run_id)
    with open (config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


# Train a model and save weights and metadata
# Example:
# $ snakemake --cores all 3_train/out/mntoha/4/hidden_size-50_weights.pt
rule train_model:
    input:
        npz_filepath = "2_process/out/{data_source}/train.npz",
        # Include configs as inputs in order to trigger save_config rule,
        # and save configs to the output folder
        process_config = "3_train/out/{data_source}/{run_id}/{run_id}_process.yaml",
        train_config = "3_train/out/{data_source}/{run_id}/{run_id}_train.yaml"
    output:
        weights_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_weights.pt",
        metadata_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_metadata.npz"
    params:
        # Read in the run and model-specific config
        config = lambda wildcards: read_train_config(
            wildcards.data_source,
            wildcards.run_id
        ),
        # Pass the run_id and model_id wildcards as params so that they can be
        # saved into metadata easily
        run_id = lambda wildcards: wildcards.run_id,
        model_id = lambda wildcards: wildcards.model_id
    # Change this number of threads as needed. It acts as a maximum.
    threads:
        80
    script:
        "3_train/src/train.py"

# Summarize all trained models in a directory using their metadata
# The parent directory could be a data_source, e.g., mntoha, or a specific run
# that produced many models, e.g., mntoha/5
rule summarize_trained_models:
    # No inputs because we don't want this rule to trigger the train_model rule
    output: 
        summary_filepath = "3_train/out/{parent_directory}/summary.csv"
    script:
        "3_train/src/summarize.py"

