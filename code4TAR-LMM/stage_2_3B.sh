
set -x

MODEL_PATH= # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/stage2_config.yaml \
    data.train_files= \
    data.val_files= \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.experiment_name=stage2 \
    trainer.n_gpus_per_node=8
