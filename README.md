This repository contains supplementary material for the conference paper
"Engineering Uniform Sampling of Graphs with a Prescribed Power-law Degree Sequence"
scheduled for presentation at ALENEX2022.

# Compilation
```
sudo apt install gcc-10 g++-10 build-essential cmake libboost-dev

git submodule update --init --recursive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

# Usage
There are two basic modes of operation; either a degree sequence is produced by the generator itself or externally provided:

## Feeding external degree sequence
A degree sequence can be provided via STDIN or a file. In each case a degree sequence with n nodes 
consists of n lines where each line contains a single number from [0, n). Observe that the sequence
is sorted decreasingly, i.e. node 0 will be the node with largest degree.

First we will generate a degree sequence of a three-star:
```bash
cd build
echo "3\n1\n1\n1\n" > degrees
```

And then produce the graph
```bash
./generator -i degrees          # either read directly
cat degrees | ./generator -i -  # or via STDIN 
```

Observe that a three-star is a threshold graph, i.e. there exists only one topology an the generator
will always output the following METIS file via STDERR:

```text
4 3
1 2 3
0
0
0
```

The first line contains the number of nodes followed by the number of edges; nodes are indexed [0...n).
Each following line i=1,2,... contains the neighbors of the (i-1)-th node. 

Alternatively, you can store the graph directly into a file:
```bash
./generator -i degrees -o threestar.metis
```

## Generating a power-law degree sequence
In order to let the generator sample a random power-law degree sequence, you need to supply the
number of nodes (`-n`) and the power-law exponent (`-g`); you may additionally constrain the minimal degree (`-a`) and maximal degree (`-b`).
```bash
./generator -n 1000 -a 2 -g 2.9 -o graph_with_1000_nodes.metis
```