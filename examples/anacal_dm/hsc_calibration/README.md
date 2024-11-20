# Setup Butler
```
sh ./butler_setup.sh
```

# Run with bps (in torque system)
```
bps submit bps_torque.yaml

```

# Run Locally
```
pipetask run -b ./ -j 1 -i skymaps -o sim -p ./shear_config.yaml -d "skymap='hsc_sim' AND tract=0 AND patch in (0) AND band in ('g', 'r', 'i', 'z', 'y')" --register-dataset-types --skip-existing --clobber-outputs
```
