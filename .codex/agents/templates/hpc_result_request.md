# HPC result request: <task_id>

Please run <notebook path> through Open OnDemand.

- Working directory: /home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/
- Environment: .venv, Python 3.12, required GPU/RAM/wall time:
- Verify cells/config:
- Expected outputs:
- Pass indicators:
- Failure indicators:

After the run, please return:
1. executed notebook or exported outputs/error log;
2. run configuration JSON;
3. summary CSV/JSON;
4. QA figures/overlays;
5. traceback if present;
6. GPU model, requested RAM and wall time;
7. observed peak GPU and host memory where available;
8. job ID and bulk-output path;
9. path, size, version and SHA-256 for large NIfTI/checkpoint outputs.

Copy compact evidence to TestProject/HPC/HPC_results/<task_id>/<run_id>/. The task remains pending until the bundle is reviewed and receives PASS, RETRY or BLOCKED.
