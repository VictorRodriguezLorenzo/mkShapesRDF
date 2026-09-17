import subprocess
from pathlib import Path
import os
import shutil
from copy import deepcopy


class BatchSubmission:
    DEFAULT_RUNNER_VARIABLES = (
        "samples",
        "aliases",
        "variables",
        "cuts",
        "nuisances",
        "lumi",
    )

    @staticmethod
    def _select_sample_config(config, sample_name):
        """Return only configuration entries that apply to ``sample_name``.

        Aliases and nuisances are the two batch configuration collections whose
        entries may be restricted with a ``samples`` field.  Keeping every
        entry (and every key in a nuisance's sample map) made each generated
        ``script.py`` carry configuration for the complete production.
        """
        selected = {}
        for name, value in config.items():
            if not isinstance(value, dict) or "samples" not in value:
                selected[name] = deepcopy(value)
                continue

            sample_spec = value["samples"]
            if sample_name not in sample_spec:
                continue

            selected[name] = deepcopy(value)
            if isinstance(sample_spec, dict):
                selected[name]["samples"] = {
                    sample_name: deepcopy(sample_spec[sample_name])
                }
            elif isinstance(sample_spec, (list, tuple, set)):
                selected[name]["samples"] = [sample_name]

        return selected

    def _batch_value(self, variable, sample_name):
        """Get a batch variable, reducing sample-aware collections per job."""
        value = self.d[variable]
        if variable in ("aliases", "nuisances"):
            value = self._select_sample_config(value, sample_name)
        if variable == "nuisances":
            for nuisance in value.values():
                for folder_key in ("folderUp", "folderDown"):
                    folders = nuisance.get(folder_key)
                    if isinstance(folders, dict):
                        nuisance[folder_key] = folders[sample_name]
        return value

    def _job_variables(self):
        """Return only the globals consumed by the selected runner.

        The bundled runner has a well-defined six-variable interface.  Older
        configuration templates made ``batchVars`` a slice of ``varsToKeep``,
        which also copied plotting, datacard, and submission-only data into
        every job.  A custom runner remains governed by ``batchVars`` because
        its required globals cannot be inferred here.
        """
        default_runner = Path(__file__).with_name("runner.py").resolve()
        if Path(self.runnerPath).resolve() == default_runner:
            return self.DEFAULT_RUNNER_VARIABLES
        return self.batchVars

    @staticmethod
    def resubmitJobs(batchFolder, tag, samples, dryRun, queue):
        """
        Resubmit failed jobs and rename the old error file to err-1.txt
        Args:
            batchFolder (string): path to the batch folder
            tag (string): string used to tag the configuration
            samples (list of strings): samples to be resubmitted in the form of ['DY_0', ...]
        """

        # Path(f'{self.batchFolder}/{self.tag}/{sampleName}_{str(i)}').mkdir(parents=True, exist_ok=False)
        for sample in samples:
            if os.path.exists(f"{batchFolder}/{tag}/{sample}/err.txt"):
                os.rename(
                    f"{batchFolder}/{tag}/{sample}/err.txt",
                    f"{batchFolder}/{tag}/{sample}/err-1.txt",
                )
        with open(f"{batchFolder}/{tag}/submit.jdl") as file:
            txt = file.read()
        lines = txt.split("\n")
        line = list(filter(lambda k: k.startswith("queue"), lines))[0]
        jobflavour = list(filter(lambda k: k.startswith("+JobFlavour"), lines))[0]
        lines[lines.index(line)] = f'queue 1 Folder in {", ".join(samples)}\n '
        lines[lines.index(jobflavour)] = f'+JobFlavour = "{queue}"\n '
        with open(f"{batchFolder}/{tag}/submit.jdl", "w") as file:
            file.write("\n".join(lines))

        if dryRun != 1:
            process = subprocess.Popen(
                f"cd {batchFolder}/{tag}; condor_submit submit.jdl; cd -", shell=True
            )
            process.wait()

    def __init__(
        self,
        folder,
        outputPath,
        batchFolder,
        headersPath,
        runnerPath,
        tag,
        samples,
        d,
        batchVars,
        jdlconfigfile,
    ):
        self.project_folder = folder
        self.outputPath = outputPath
        self.batchFolder = batchFolder
        self.headersPath = headersPath
        self.runnerPath = runnerPath
        self.tag = tag

        self.samples = samples
        self.d = d
        self.batchVars = batchVars
        self.jdlconfigfile = jdlconfigfile

        self.folders = []

    def createBatch(self, sample):
        # 1. create submission folder
        # 2. create executable python file
        # 3. create bash file
        # 4. create condor file
        # 5. append condor file to submit files

        # submission folder
        sampleName = sample[0]
        i = sample[3]
        try:
            Path(f"{self.batchFolder}/{self.tag}/{sampleName}_{str(i)}").mkdir(
                parents=True, exist_ok=False
            )
        except:  # noqa E722
            print("Error creating condor folder!")
        self.folders.append(f"{sampleName}_{str(i)}")
        # python file

        txtpy = "from collections import OrderedDict\n"

        # ``sample`` is already the single chunk produced by splitSamples.
        _samples = [sample]

        txtpy += f"samples = {str(_samples)}\n"

        for var in self._job_variables():
            _var = var
            if not isinstance(var, str):
                _var = var[0]

            if _var == "samples":
                continue
            value = self._batch_value(_var, sampleName)
            if isinstance(value, int) or isinstance(value, float):
                txtpy += f"{_var} = {value}\n"
            else:
                txtpy += f"{_var} = {str(value)}\n"

        with open(
            f"{self.batchFolder}/{self.tag}/{sampleName}_{str(i)}/script.py", "w"
        ) as f:
            f.write(txtpy)

    def createBatches(self):
        try:
            print("Removing dir:", os.path.abspath(f"{self.batchFolder}/{self.tag}"))
            shutil.rmtree(os.path.abspath(f"{self.batchFolder}/{self.tag}"))
        except Exception as e:
            print("Error removing directory", e)

        for sample in self.samples:
            self.createBatch(sample)

    def submit(self, dryRun=0, queue="workday"):

        txtsh = ""
        use_jdlconfigfile = self.jdlconfigfile != ""

        if use_jdlconfigfile:
            try:
                print("Opening jdlconfigfile")
                print(self.project_folder + "/" + self.jdlconfigfile)
                exec(
                    open(self.project_folder + "/" + self.jdlconfigfile).read(),
                    globals(),
                )
            except Exception as e:
                print('could not parse jdlconfigfile "', self.jdlconfigfile, '"\n', e)
                use_jdlconfigfile = False

        if use_jdlconfigfile:
            txtsh += "\n".join(executable)
        else:
            with open(os.environ["STARTPATH"]) as file:
                txtsh += file.read()

            mE = self.d.get("mountEOS", [])
            for line in mE:
                txtsh += line

            runnerScriptFilename = self.runnerPath.split("/")[-1]
            txtsh += f"time python {runnerScriptFilename}\n"

            outputFileTrunc = ".".join(self.d["outputFile"].split(".")[:-1])

            print("\n\nReal output path:", os.path.realpath(self.outputPath), "\n\n")

            if os.path.realpath(self.outputPath).startswith("/eos"):
                # eos is not supported -> use xrdcp
                fullOutfile = f"{os.path.realpath(self.outputPath)}/"
            else:
                fullOutfile = f"{self.outputPath}/"

            fullOutfile += f"{outputFileTrunc}__ALL__" + "${1}.root"
            txtsh += f"cp output.root {fullOutfile}\n"
            txtsh += "rm output.root\n"
            txtsh += "rm script.py\n"

        # write the run.sh file
        with open(f"{self.batchFolder}/{self.tag}/run.sh", "w") as file:
            file.write(txtsh)
        # make it executable
        process = subprocess.Popen(
            f"chmod +x {self.batchFolder}/{self.tag}/run.sh", shell=True
        )
        process.wait()

        txtjdl = "universe = vanilla \n"
        txtjdl += "executable = run.sh\n"
        txtjdl += "arguments = $(Folder)\n"

        txtjdl += "should_transfer_files = YES\n"

        if use_jdlconfigfile:

            for key in jdl_dict:
                if jdl_dict[key] != "":

                    txtjdl += key + " = " + jdl_dict[key] + "\n"
        else:

            txtjdl += f"transfer_input_files = $(Folder)/script.py, {self.headersPath}, {self.runnerPath}\n"

        txtjdl += "output = $(Folder)/out.txt\n"
        txtjdl += "error  = $(Folder)/err.txt\n"
        txtjdl += "log    = $(Folder)/log.txt\n"

        txtjdl += "request_cpus   = 1\n"
        txtjdl += f'+JobFlavour = "{queue}"\n'

        txtjdl += f'queue 1 Folder in {", ".join(self.folders)}\n'
        with open(f"{self.batchFolder}/{self.tag}/submit.jdl", "w") as file:
            file.write(txtjdl)

        condor_args = ""
        if dryRun != 1:

            if use_jdlconfigfile:
                condor_args += " ".join(condor_config)

            proc_command = f"cd {self.batchFolder}/{self.tag}; condor_submit {condor_args} submit.jdl ; cd -"
            print(proc_command)

            process = subprocess.Popen(
                proc_command,
                shell=True,
            )
            process.wait()
