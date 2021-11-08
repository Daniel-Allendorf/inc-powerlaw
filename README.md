This repository contains supplementary material for the conference paper
"Engineering Uniform Sampling of Graphs with a Prescribed Power-law Degree Sequence"
scheduled for presentation at ALENEX2022.

# Reproducibility
To reproduce our experiments, feel free to run the following code. Please note that the experiments
will run for roughly 2 weeks on a 32 core machine.

```
apt install gcc-10 g++-10 build-essential cmake libboost-dev
cd automated-build-and-benchmark
./build_and_run_benchmarks
```

If you get an error that 'networkit' cannot be found by python, make sure that it is installed:

```
pip3 install networkit
```