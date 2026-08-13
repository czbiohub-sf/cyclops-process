NOTE: Currently in this directory there are two separate nextflow implementations. The documentation in this README refers exclusively to the implementation encapsulated by `iss.nf` and `iss.config`. It does its best not to share any configuration files with the other `main.nf` implementation so as to reduce confusion. If any routines are shared, they are done so by replicating the routine inside of the new flow rather than referencing the routine in the older one.

The justification for the existence of two implementations is that the latter was a first pass using Claudecoded assistance whereas the former represents a concerted effort to bring the LLM-generated code more in line with the established habits and patterns of the OPS Pipeline.

## Usage

### Before Start: Setting Up Environment Variables
- Three environment variables are necesssary for the OPS pipeline to function properly:
    - OPS_OUTPUT_BASE_DIR
        - A string filepath to the parent directory into which an OPS project directory will exist. Most steps in the pipeline access this variable via the OPSDataset python object and write stuff referentially through this root.
    - OPS_CONFIGS_DIR
        - A string filepath to the directory with configuration information about the pipeline. In general these configs contain python keyword arguments and other python context written as YAML.
        - The path to the root directory then matches to a file based on the `--experiment` argument being passed in.
    - OPS_EXP_CONFIG_FILE
        - String filepath to the specific config file that is to be used for this run. Generally used as an override or substitute for missing `OPS_CONFIGS_DIR`. Anecdotally, this env var seems a bit flaky in the nextflow world.

#### Setting up ENV Vars:
Nextflow looks at the `env` object and can override local env vars with its contents. As such, it is most advised that environment variables go in here.

In `iss.config` Point your environment variables at your working location.

This folder substructure inside of `/mydata/` is not dogmatic. Feel free to customize to your comfort.
```
env {
    OPS_OUTPUT_BASE_DIR = "/path/to/your/workspace/test_ops_data/"
    OPS_CONFIGS_DIR = "/path/to/your/workspace/test_ops_data/configs/"
    OPS_EXP_CONFIG_FILE = "/path/to/your/workspace/test_ops_data/configs/ops0094_20251217_mark_config.yaml"
}
```


### Starting a Nextflow run:
- In the `./nextflow/`* directory a nextflow run can be started with the following command:
    - `nextflow run iss.nf -c iss.config -params-file nextflow_ops_args.yaml -with-dag -with-report`
    - A breakdown of what each section does:
        - `nextflow run iss.nf`
            - This command starts the workflow. It defaults without an argument to target `main.nf` but we override the argument here with `iss.nf` to specify the workflow we want to run.
        - `-c iss.config`
            - Designates a `.config` file to pull. More documentation on what the config file is concerned with is documented here: https://www.nextflow.io/docs/latest/config.html#configuration-files
            - The naming convention here is arbitrarily to match the config to the name of the workflow. This isn't enforced inside of Nextflow in anyway and is just a naive orginizing strategy.
        - `-params-file nextflow_ops_args.yaml`
            - The parameters file is a way of expressing variables to inject into the configuration file. This option for introducing parameters was chosen because this project currently heavily uses YAML.
    - This command will occupy the current terminal so we are currently looking at wrapping it in a `slurm` command so that it can asynchronously manage scheduling.
        - Example Slurm command:
        ```
        sbatch --job-name=nf-iss --output=logs/nf-iss-%j.log --partition=cpu --cpus-per-task=2 --mem=8G --time=12:00:00 --wrap="source /path/to/cyclops-monorepo/.venv/bin/activate && nextflow run iss.nf -c iss.config -params-file nextflow_ops_args.yaml"
        ```
- *: Nextflow is reliant upon the directory it's called in for initializing a bunch of underlying filepaths so for now until a more generalised solution is figured out it is best to invoke calls only from this directory.

#### Resuming a failed run:

- You can resume failed runs by passing `-resume ${FAILED_SESSION_ID}` to your Nextflow command.
    - If a run fails, you will need to find the session ID for the run. Nextflow assigns runs a human-friendly name, e.g., `infallible_neumann`, and a session UUID like `024ace66-81d9-4549-ab6c-ada96d3a6d29`. The human-friendly name is at the top of your HTML report. To find the session ID, run `cat .nextflow/history | grep ${HUMAN_FRIENDLY_NAME}`. The session ID is the long UUID string in that line.
    - Once you have the session ID, you can resume the run with `nextflow run iss.nf -c iss.config -params-file nextflow_ops_args.yaml -resume ${FAILED_SESSION_ID} -with-dag -with-report`. You can also slot this command into your `sbatch --wrap` to resume as a Slurm job.

### Under the Hood:
- When we issue a Nextflow run from the CLI, a lot of behaviour is being encapsulated under the call. This section serves to provide a more in depth walk through of the underlying behaviour so as to better communicate the overall architecture and the relationships between the active components.
- ISS post-stitch: after registration, `merge_spots_base_calling` fuses the warp, spot detection, and base calling into one per-well job, keeping the registered array in shared memory (`finalize_iss_registration` composes transforms only, via `skip_apply_transforms`). This replaces the older separate `detect_spots` → `base_calling` stages.

### Some Tensions / Speculations
- Nextflow likes to have finely grained management over the underlying filesystem and artifacts it interacts with. It does a lot of this by building symlinks in the `work`. However, the standard operating procedure for the OPS pipeline is to manage this context inside of python. Whether or not this is the best way to go about things remains to be seen but in present state it is effective and fault tolerant.

### Cleaning Up
In general because the workflow writes a lot of content to storage, I have found it most effective to delete my workspace at the end of every test workflow run and regenerate it. I've done this with a `rm --rf <path to my dir> && mkdir <path to my dir>` shell command.

It is heavily advised that you do something similar. This delete can take on the order of low minutes due to the size of the dataset (approx 100s of GB).

## Developing
### Getting Started

The underlying workflow behaviour relies heavily on the file system. As such, in order to make dev work nondisruptive it is heavily recommended that you point with READ level access at incoming data and write to a workspace inside of `/path/to/your/workspace`.

We use `ops0094_20251217_mark` as the default experiment for doing dev work against as it is a copy of a prod dataset specifically for testing.


### Writing Nextflow Tasks

For the sake of this discussion, the fundamental generalization being made is that one python callable corresponds to one conceptual step which corresponds to one Nextflow “process”. For example, given the python function `do_thing()`, there ought to exist an equivalent Nextflow process roughly declared as:
```
process nextflow_do_thing {
	script:
	“””
		python -m do_thing()
	“””
}
```

In creating a port from the existing `PipelineRunner` framework to Nextflow, the `runner.run()` calls in `orchestration.py` correspond mostly 1:1 to a python call. That is, the function that is wrapped by `runner.run()` represents the underlying python that is desired for execution.
	- There are a couple edge cases where fan-out constructs need to be created for python calls that spin out slurm subjobs. This execution pattern is programmatically valid  but it means that the code exits the scope of Nextflow’s purview. Meaning that the code will run but it’s significantly antithetical to best practices.

#### Porting Functionality.

1. Identify the python callable that needs to be executed in Nextflow.
2. In `dispatch_cli.py` add the string identifier and callable to the lookup object.
	2 a. Do imports as necessary
3. In `convert.nf` Create a process with a similarly shaped name to the relevant python callable
    3 a. Confirm that the python callable does not create phantom slurm jobs on the fly for any of its work. If it does, follow the below substeps. If not, skip to 3 b.
        3 a i. Because Nextflow has no awareness of phantom slurm jobs, the unit of parallelization that is being passed as a slurm command needs to be bubbled up and extracted such that it is a fan out of child jobs all descendants of one setup job that distributes the work. The first step is to identify the functional unit of slurm parallelization.
        3 a ii. Extract, copy, and/or implement the surrounding distribution and preparation logic as a setup task. This setup task should ingest from the same upstream command and dispatch to a fan out of child commands containing the parallelization and any parameters that need to be passed on.
        3 a iii. Taking the unit of parallelization, confirm that there are no nested slurm commands inside of it. Multithreading, multiprocessing, or other well defined fan out patterns are acceptible.
        3 a iv. Wire the setup task to the fanout tasks and write a Nextflow task to collect results if needed. Continue with step 3 b.
	3 b. Wire up the process as necessary
	3 c. Emit process outputs as necessary
	3 d. Set up python keyword args 
4. In `nextflow_ops_args.yaml` configure slurm parameters
5.  In `iss.config` using the `withName` keyword, generate a config specification for the corresponding Nextflow process 
	5 a. Reference params from the YAML as necessary
6. In `iss.nf` import the newly made Nextflow process
7. Invoke it as necessary
### Changing Experiment Sources:

To point at a different source:

  1. OpsDataset (cyclops_utils/src/cyclops_utils/data/experiment.py, line 33) — change iss_tif_dir (and lc_dragonfly_dir if using live cell data) to point at the alternate
  location. These are the only two hardcoded source paths.
  2. OPS_EXP_CONFIG_FILE — if the source data lives under a different configs directory, update this as well (already noted as a TODO).
  
### Changing Experiment Destinations:

When switching experiments, the checklist is:
  1. nextflow_ops_args.yaml line 1: update experiment
  2. iss.config: update the OPS_EXP_CONFIG_FILE fallback (or set it as an env var)
  3. Optionally set OPS_OUTPUT_BASE_DIR if targeting a different destination root

### How we pass arguments down in Nextflow

 nextflow_ops_args.yaml
    └─ params.processes.<step>.python_kwargs (Map)
         └─ iss.config withName block: ext.kwargs = params.processes.<step>.python_kwargs
              └─ convert.nf kwargsToArgs(task.ext.kwargs) → "--experiment foo --other bar"                
                   └─ scratch_cli.py --experiment foo --other bar
                        └─ Python callable(experiment="foo", other="bar")

