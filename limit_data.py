import numpy as np

command = """python train.py --data_dir=/home/storage/sensorium --output_dir=/home/storage/runs/limit_data_2/viv1t/{output_dir} --ds_mode=2 --core=vivit --core_behavior_mode=2 --core_use_causal_attention --readout=gaussian --output_mode=1 --schedule_free --compile --batch_size=1 --wandb=limit_data_2 --limit_data={limit_data} --limit_neurons={limit_neurons} --seed={seed} --clear_output_dir"""

seed = 1234
max_sample = 350
run_id = 1
for limit_data in 0.01 * np.linspace(10, 100, 10):
    limit_data = int(limit_data * max_sample)
    for limit_neurons in [10, 50, 100, 200, 300, 400, 500, 600]:
        output_dir = (
            f"{run_id:03d}_{limit_data}data_{limit_neurons:04d}neuron_{seed:04d}seed"
        )
        print(
            command.format(
                limit_data=limit_data,
                limit_neurons=limit_neurons,
                seed=seed,
                output_dir=output_dir,
            )
        )
        run_id += 1
