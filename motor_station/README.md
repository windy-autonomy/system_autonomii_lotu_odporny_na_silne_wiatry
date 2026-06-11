# OpenScale Recorder and Analysis

Desktop tools for reading a SparkFun OpenScale load cell, recording raw sessions, and filtering test data.

## What is in this repo

- `open_scale.py` - Tkinter recorder with connect/start/stop controls and a live plot
- `filter_openscale.py` - offline filter and plot tool for recorded CSV sessions
- `openscale_processing.py` - shared parsing, filtering, and impulse calculations
- `plot.py` - compatibility wrapper for the offline plot/filter workflow

## Installation

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

If `pyserial` is missing, the recorder will show an error until it is installed.

## Live recording

Start the recorder:

```bash
python3 open_scale.py --port /dev/ttyUSB0 --baudrate 115200 --output-dir recordings
```

In the app:

- enter the serial port
- click `Connect`
- click `Start` to begin a session
- click `Stop` to end the session

The app saves a raw CSV session file with the original serial text and parsed readings.

## Offline filtering

Filter and plot a raw session:

```bash
python3 filter_openscale.py recordings/first_test.csv
```

Optional arguments:

- `--window-size N` - Hampel and rolling-median window size
- `--sigma X` - outlier rejection threshold
- `--tau X` - low-pass smoothing time constant
- `--propellant-mass-kg M` - compute specific impulse in seconds
- `--no-show` - skip opening the plot window

The script writes:

- a cleaned CSV in `processed/`
- a comparison plot PNG in `processed/`
- printed total impulse in `N*s`

Example:

```bash
python3 filter_openscale.py recordings/first_test.csv
```

## Data format

Raw recorder CSV files contain:

- timestamps in Unix seconds and UTC
- elapsed time from the start of the session
- raw serial text
- parsed numeric value
- source unit
- numeric flag

The offline filter converts the recorded readings to force in Newtons and then:

- rejects spikes with a Hampel filter
- smooths with a rolling median plus low-pass EMA
- prints total impulse
