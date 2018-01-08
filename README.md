# Ratatouille
Optimizing Network Performance for Large-Scale Distributed Machine Learning

The Parameter Server (PS) framework is widely used to train machine learning (ML) models in parallel. It tackles the big data problem by having worker nodes performing data-parallel computation, and having server nodes maintaining globally shared parameters. When training big models, worker
nodes frequently pull parameters from server nodes and push updates to server nodes, resulting in
high communication overhead. Our investigations showed that modern distributed ML applications
could spend up to 5.7 times more time on communication than computation. To address this problem,
we propose a novel communication layer for the PS framework using self-imposed sparsity. First, we
introduce an update-centric communication model to exchange data between worker and server
nodes via two operations: broadcast and push. With this model, only updates are transmitted over
network. Second, we develop a dynamic value-bounded filter to reduce network traffic by selectively
dropping updates before transmission. With this filter, server and worker nodes only need to transmit
a small portion of updates during each broadcast and push operations. Theoretical analysis shows
that our approach could reduce the network traffic and communication time significantly with
convenience guarantees. Existing extensive performance evaluations showed that PF could speed up
popular distributed ML applications by a factor of up to 4.3, compared to the conventional PS
framework. This technique could be used for large-scale distributed ML in Azure.
