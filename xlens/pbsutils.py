import parsl
from parsl.providers import TorqueProvider
from parsl.executors import HighThroughputExecutor
from parsl.config import Config
from parsl.addresses import address_by_hostname


class PBS(object):
    def __init__(
        self,
        nodes,
        cores_per_node,
        mem_per_node,
        walltime,
        queue,
        provider_options,

    ):
        self.nodes = nodes
        self.cores_per_node = cores_per_node
        self.mem_per_node = mem_per_node
        self.walltime = walltime
        self.provider_options = provider_options
        self.queue = queue

    def get_config(self):
        # Set scheduler options for Torque, specifying nodes, cores per node, and memory
        scheduler_options = \
            f"#PBS -l nodes={self.nodes}:ppn={self.cores_per_node},mem={self.mem_per_node}gb"

        # Use TorqueProvider for PBS system
        torque_provider = TorqueProvider(
            nodes_per_block=self.nodes,
            init_blocks=1,  # Start with 1 block
            max_blocks=10,  # Set maximum blocks, as needed
            walltime=self.walltime,
            account=self.provider_options.get('account', None),
            queue=self.queue,
            scheduler_options=scheduler_options,  # Pass the scheduler options here
            worker_init=self.provider_options.get('worker_init', '')
        )

        # Create HighThroughputExecutor using TorqueProvider
        executor = HighThroughputExecutor(
            label="htex_pbs",
            address=address_by_hostname(),
            provider=torque_provider,
        )

        # Return Parsl configuration
        return Config(executors=[executor])


# Example instantiation (could be used within your script)
pbs_site = PBS(
    nodes=5,
    cores_per_node=5,
    mem_per_node=35,
    walltime="24:00:00",
    queue="small",
    provider_options={
        "account": "xiangchong.li",
        "scheduler_options": "",
        "worker_init": ""
    }
)

# Load Parsl configuration
parsl.load(pbs_site.get_config())
