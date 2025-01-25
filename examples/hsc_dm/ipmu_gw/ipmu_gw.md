# Example

## Butler

On GW@IPMU, we can use the butler here:
```shell
/work/xiangchong.li/work/hsc_s23b_sim/sim_cosmos/simulation
```

If you wish to not using existing butler, you can setup LSST DM butler using
the skymaps defined in [here](./skymap.py), with the following command

```shell
sh ./butler_setup.sh
```

## Simulation

Make image simulations using PSFs and noise correlation functions randomly
sampled from the survey.

```shell
pipetask run -b /work/xiangchong.li/work/hsc_s23b_sim/sim_cosmos/simulation -j 1 -i sim --output pfs_target/sim_image1 -p ./pfs_target/sim_config.yaml -d "skymap='hsc' AND tract in (16012) AND patch=34 AND band in ('g', 'r', 'i', 'z', 'y')" --register-dataset-types
```

We can change the `simulator.layout` to `grid` or `random` to change the layout
of the simulation.
Image noise can be disabled by setting `simulator.draw_image_noise=False` in
the [config file](./pfs_target/sim_config.yaml).

## DM image processing

Run DM pipeline on simulated image

```shell
pipetask run -b /work/xiangchong.li/work/hsc_s23b_sim/sim_cosmos/simulation -j 1 -i pfs_target/sim_image1 --output pfs_target/dm_catalog1 -p ./pfs_target/dm_config.yaml -d "skymap='hsc' AND tract in (16012) AND patch=34 AND band in ('g', 'r', 'i', 'z', 'y')" --register-dataset-types
```

## Parallel Processing

We can do parallel processing using the BPS system with the following command

```shell
bps submit ./pfs_target/bps_sim.yaml
```

Note, to run the BPS system on IPMU's torque server, we need to use the forked
`ctrl_bps_parsl` [here](https://github.com/mr-superonion/ctrl_bps_parsl).

To setup the BPS system, please forke, install and configure the system as
follows:

```shell
export BPS_WMS_SERVICE_CLASS=lsst.ctrl.bps.parsl.ParslService
setup -kr ctrl_bps_parsl_directory
```
