# MLSys'21 Artifacts

We document how the experiments reported in 
[Horizontally Fused Training Array: An Effective Hardware Utilization Squeezer for Training Novel Deep Learning Models](https://arxiv.org/abs/2102.02344)
can be reproduced by following the steps below.

- To measure the training throughputs:
  - On V100 and RTX6000, please follow the steps in [V100 & RTX6000 Throughput Measurement Guide](./v100_rtx6000_throughputs.md)
  - On A100, please follow the steps in [A100 Throughput Measurement Guide](./a100_throughputs.md)
  - On TPU v3, please follow the steps in [TPU v3 Throughput Measurement Guide](./tpu_v3_throughputs.md)
- To measure the hyper-parameter tuning costs of HFHT:
  - On V100, please follow the steps in [V100 HFHT Guide](./v100_hfht.md)
