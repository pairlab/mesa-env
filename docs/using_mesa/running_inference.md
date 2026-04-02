# Running Inference

The recommended way to run inference is to use the `eval_server.py` script. This script will run a websocket policy server and run rollouts on the evaluation set, outputting videos and statistics. With this approach, you can interface with your policy after writing a minimal policy server wrapper, avoiding the necessity to merge the `mesa`'s dependencies into your own.


## Running the evaluation server

To start the evaluation server, run the following command:

```bash
uv run scripts/eval_server.py \
  --exp-name <exp_name> \
  --variant-name <variant_name> \
  --eval-set-name <eval_set_name> \
  --num-rollouts-per-task <num_rollouts_per_task> \
  --controller-type <controller_type> \
  --port <port>
```

Some notable arguments:
- `--exp-name` and `--variant-name`: These optional arguments are used to determine the output directory for the evaluation results. If provided, the results will be saved in `experiments/<eval_set_name>/<exp_name>/<variant_name>/`. If not provided, the results will be saved in `experiments/<eval_set_name>/<date>/<time>/`.
- `--eval-set-name`: The name of the evaluation set to run.
- `--num-rollouts-per-task`: The number of rollouts to run per task. This will iterate through all the task variant BDDL files before repeating any.
- `--controller-type`: The type of controller to use. Valid options are `osc_pose` and `joint_pos`. If using delta actions, you'll need to add `--control-delta`.
- `--port`: The port to run the server on. This should be the same port as the one you use to run your policy server.

The sequential `eval_server.py` script is useful for debugging, when speed is not a concern. For large evaluation jobs, you should use the parallel `eval_server_parallel.py` script which has the same interface and outputs but runs rollouts in parallel.


## Start a policy server

Next, you'll need to implement a policy server that ingests observations and returns actions. Our policy server is based on the one from [openpi](https://github.com/Physical-Intelligence/openpi). To implement your own, you'll simply need to install the `openpi-client` package (found in `third_party/openpi-client` or [in the openpi repo](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client)) and implement a policy server that conforms to the `openpi_client.websocket_policy_server.WebsocketPolicyServer` interface. A sample policy server which outputs random actions is provided in `scripts/demo_policy_server.py`, which you can run with

```bash
uv run scripts/demo_policy_server.py --port <port>
```

For example, to run a policy server that outputs random delta end effector pose actions, you would run

```bash
uv run scripts/demo_policy_server.py \
  --port 8001 \
  --controller-type osc_pose \
  --control-delta
```

in one terminal and

```bash
uv run scripts/eval_server.py \
  --port 8001 \
  --eval-set-name mesa-70 \
  --num-rollouts-per-task 10 \
  --controller-type osc_pose \
  --control-delta \
  --render
```
in another.

For real models, run your model-specific websocket server and point MESA to it. For example, our policy server for pi models is provided in [our fork of the openpi repo](https://github.com/pairlab/openpi-mesa/blob/main/scripts/serve_policy.py).

For more information on running inference with our trained models, please refer to the [evaluating trained policies](../mesa_bench/evaluating_trained_policies.md) page.