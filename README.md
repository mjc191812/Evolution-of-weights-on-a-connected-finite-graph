# Evolution-of-weights-on-a-connected-finite-graph

This is the code of the paper **Evolution of weights on a connected finite graph**

## Abstract
On a connected finite graph, we propose an evolution of weights including
Ollivier's Ricci flow as a special case. During the evolution process, on each edge, the speed of change of weight is exactly the difference between the Wasserstein distance related to two probability measures and certain graph distance. Here the probability measure may be chosen as
an $\alpha$-lazy one-step random walk, an $\alpha$-lazy two-step random walk, or a general probability measure. Based on the ODE theory, we show that the initial value problem has a unique global solution.

A discrete version of the above evolution is applied to the problem of community detection. Our algorithm is based on such a discrete evolution,
where probability measures are chosen as $\alpha$-lazy one-step random walk and $\alpha$-lazy two-step random walk respectively. Note that the later measure has not been used in previous works \cite{Ni-Lin, Lai X, Bai-Lin, M-Y1}. Here, as in \cite{M-Y1}, only one surgery needs to be performed after the last iteration. Moreover, our algorithm is much easier than those of \cite{Lai X,Bai-Lin,M-Y1}, which were all based on Lin-Lu-Yau's Ricci curvature. 
