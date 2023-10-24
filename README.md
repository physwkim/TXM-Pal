# palxanes
## Analysis software for BL7C

### Installation
1) download miniconda3
2) Launch "Anaconda Prompt(miniconda3)" for windows
3) Install git
    - conda install git
4) Download palxanes repo
    - git clone https://github.com/physwkim/palxanes.git 
5) Create python environment
    - conda env create -f ./env.yaml
6) Install rust library
    - pip install lmfitrs-0.1.0-cp38-none-win_amd64.whl
11) Run main.py
    - python main.py

### Build library [Optional]
1) Install rustup
    https://rustup.rs/#
2) Download submodule
    - git submodule update --init --recursive
3) cd palxanes/lib
4) conda activate palxanes
5) Install library
    maturin develop -r
