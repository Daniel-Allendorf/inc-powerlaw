#!/usr/bin/env python3
import argparse
import networkit as nk
import time

parser = argparse.ArgumentParser(description='EdgeSwitching MCMC')
parser.add_argument("-n", "--log2minnodes", default=10, type=int)
parser.add_argument("-m", "--log2maxnodes", default=10, type=int)
parser.add_argument("-s", "--sequences", default=5, type=int)
parser.add_argument("-g", "--graphs", default=5, type=int)
parser.add_argument("-a", "--mindegree", default=1, type=int)
parser.add_argument("-e", "--gamma", default=2.88103, type=float)
args = parser.parse_args()

for num_nodes in (int(2**i) for i in range(args.log2minnodes, args.log2maxnodes+1)):
    max_deg = num_nodes ** (1 / (args.gamma - 1))
    seq = nk.generators.PowerlawDegreeSequence(args.mindegree, max_deg, -args.gamma)
    seq.run()

    for seq_index in range(args.sequences):
        degseq = seq.getDegreeSequence(num_nodes)

        for grp_index in range(args.graphs):
            start_time = time.time()
            G = nk.generators.EdgeSwitchingMarkovChainGenerator(degseq, True).generate()
            elapsed = time.time() - start_time
            print(",".join(map(str, [G.numberOfNodes(), G.numberOfEdges(), args.gamma, args.mindegree, max_deg, 10, 1000*elapsed])))
