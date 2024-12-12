import time
from .template import template_string
from string import Template
import subprocess

gw_node_list = [
    "hpcs005",
    "hpcs006",
    "hpcs007",
    "hpcs008",
    "hpcs009",
    "hpcs010",
    "hpcs011",
    "hpcs012",
    "hpcs013",
    "hpcs014",
    "hpcs015",
    "hpcs016",
    "hpcs017",
    "hpcs018",
    "hpcs019",
    "hpcs020",
    "hpcs021",
    "hpcs022",
    "hpcs023",
    "hpcs024",
]


def submit_job(configs_dict):
    script_filename = "./submit/sh/%s.sh" % (configs_dict["jobname"])
    submit_script = Template(template_string).substitute(**configs_dict)

    with open(script_filename, "w") as f:
        f.write(submit_script)
    # subprocess.run(
    #     ["qsub", script_filename],
    #     capture_output=True,
    #     text=True,
    #     check=True
    # )
    # time.sleep(10)
    return
