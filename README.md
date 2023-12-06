# graphclustering
A balanced graph clustering algorithm

该代码中的graph clustering算法属于流失聚类算法，该算法本质上为Nishimura等人所提出的reLDG算法实现，为了加快大规模集合交集的求解过程，引入了MinHash方法，并且加入了q-fraction节点筛选过程。
reLDG算法的参考文献：
Nishimura J, Ugander J. Restreaming graph partitioning: simple versatile algorithms for advanced balancing. KDD 2013: 1106-1114.

q-fraction节点筛选过程的参考文献：
Shang H, Shi X, Jiang B. Network A/B Testing: Nonparametric Statistical Significance Test Based on Cluster-Level Permutation. Journal of Data Science, 2023, 21(3):523-537.

为了便于理解本文算法原理和所需解决的问题，整理出一个PDF文档，供大家参考。
