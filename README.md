# DAS-realtime

A Python package to process distributed acousting sensing real-time data streams for earthquake monitoring and early warning. This package can also stream selected channels using a [PyEarthworm](https://github.com/Boritech-Solutions/PyEarthworm) paradigm.

## Installation

In order to install DAS-realtime, you need to have first create a conda environment and install all the required packages. Run the following commands after cloning the repo.

```
conda env create -f environment.yml

git submodule update --init --recursive external/EQNet

cd external/EQNet/

pip install -r requirements.txt

pip install obspy fastapi

```

In addition, you will need to have PyEarthworm installed as well as Earthworm. Follow the installation guide within the [PyEarthworm](https://github.com/Boritech-Solutions/PyEarthworm) repository.

# Citation

If you are using this software for your research, please, cite the associated publication:

Biondi, E., Tepp, G., Yu, E., Saunders, J. K., Yartsev, V., Black, M., Watkins, M., Bhaskaran, A., Bhadha, R., Zhan, Z., & Husker, A. L. (2025). Real-time processing of distributed acoustic sensing data for earthquake monitoring operations. Manuscript under review at Seismological Research Letters.