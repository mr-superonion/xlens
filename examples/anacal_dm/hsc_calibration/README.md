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


# Shear Distortion Mode


Note that there are three options in each redshift bin
+ 0: g=-0.02;
+ 1: g=0.02;
+ 2: g=0.00

For example, number of redshift bins is 4, (nz_bins = [0., 0.5, 1.0, 1.5,
2.0]), if mode = 7 which in ternary is "0021" --- meaning that the shear is
(-0.02, -0.02, 0.00, 0.02) in each bin, respectively.

