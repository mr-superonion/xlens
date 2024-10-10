import os
import subprocess


def test_pipetask_run():
    # setup butler
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)
    command = ["sh", os.path.join(this_dir, "butler_setup.sh")]
    subprocess.run(command, capture_output=False, text=False)

    # run pipeline tasks
    command = [
        "pipetask",
        "run",
        "-b",
        this_dir,
        "-j",
        "1",
        "-i",
        "skymaps",
        "-o",
        "run",
        "-p",
        os.path.join(this_dir, "shear_config.yaml"),
        "-d",
        "skymap='hsc_sim' AND tract=0 AND patch=0 AND band='i'",
        "--register-dataset-types",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # Assert that the command executed successfully
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"


if __name__ == "__main__":
    test_pipetask_run()
