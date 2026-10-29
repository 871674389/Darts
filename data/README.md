# Datasets

The datasets are **not** shipped with this repository. All of them are public
benchmarks; download them from their original sources and arrange each one in
the layout shown below (paths are hard-wired in `config_files/config_<DS>.py`).

Each series is stored as NumPy arrays of shape `(T, num_features)` (float), and
the anomaly ground truth is stored as time intervals (start,end rows) or a
per-timestamp label column, exactly as the corresponding config expects.

## Expected layout

```
data/
├── SWAT/                      # Secure Water Treatment  (51 sensors)
│   ├── train.npy
│   ├── test.npy
│   └── SWAT_Time.csv          # anomaly intervals; downsampling_fre = 60
├── WADI/                      # Water Distribution      (118 sensors)
│   ├── train.npy
│   ├── test.npy
│   └── labels.csv             # anomaly intervals; downsampling_fre = 300
├── MSL/                       # Mars Science Laboratory (55 channels)
│   ├── train.npy
│   ├── test.npy
│   └── MSL_test_anomaly_intervals.csv     # downsampling_fre = 1
├── SMAP/                      # Soil Moisture Active Passive (25 channels)
│   ├── train.npy
│   ├── test.npy
│   └── SMAP_test_anomaly_intervals.csv    # downsampling_fre = 1
├── PSM/                       # Pooled Server Metrics   (25 features)
│   ├── PSM_train.npy
│   ├── PSM_test.npy
│   └── PSM_test_anomaly_intervals.csv
└── SMD/                       # Server Machine Dataset  (38 features, 28 machines)
    └── machine-1-1/           # one sub-folder per machine (select via DUALV4_SMD_MACHINE)
        ├── train.npy
        ├── test.npy
        └── label.csv
```

## Where to download

| Dataset | Source |
|---|---|
| SWaT / WADI | iTrust, SUTD — request access at <https://itrust.sutd.edu.sg/itrust-labs_datasets/> |
| SMAP / MSL  | NASA, via the Telemanom repo <https://github.com/khundman/telemanom> |
| SMD         | OmniAnomaly <https://github.com/NetManAIOps/OmniAnomaly> |
| PSM         | eBay, via the Anomaly-Transformer repo <https://github.com/thuml/Anomaly-Transformer> |

## Preprocessing helpers

Some conversion / cleaning utilities used while preparing SMD and PSM live in
`scripts/data_prep/`. They read a copy of the raw `Machine` folder; point them
at it with an environment variable, e.g.:

```bash
export DARTS_IMD_MACHINE_DIR=/path/to/IMDiffusion-master/data/Machine
python scripts/data_prep/prep_smd_clean.py
```

These helpers are provided for transparency/reproducibility; you can also build
the `.npy`/label files with your own pipeline as long as the layout above holds.
