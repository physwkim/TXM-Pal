# TXM-Pal
## TXM Analysis software

### Installation
1) download miniconda3
2) Launch "Anaconda Prompt(miniconda3)" for windows
3) Install git
    - conda install git
4) Download TXM-Pal repo
    - cd \
    - mkdir codes
    - cd codes
    - git clone https://github.com/physwkim/TXM-Pal.git
5) Create python environment
    - conda env create -f ./env.yaml
6) Activate environment
    - conda activate txm-pal
6) Install rust library (On windows)
    - cd wheels
    - pip install lmfitrs-0.1.6-cp312-none-win_amd64.whl
    - cd ..
11) Run main.py
    - python main.py

### Build library [Optional]
1) Install rustup
    https://rustup.rs/#
2) Download submodule
    - git submodule update --init --recursive
3) cd TXM-Pal/lib
4) conda activate txm-pal
5) Install library
    maturin develop -r
