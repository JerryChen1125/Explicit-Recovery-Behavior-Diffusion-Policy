"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""




import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
            dir=str(output_dir),
         
           
            settings=wandb.Settings(start_method='fork')
        )
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.training.seed=42
   
    cls = hydra.utils.get_class(cfg._target_)
    cfg.dataloader.batch_size=6
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    cfg.dataloader.batch_size=30
    dataset= dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    # get policy from workspace
    
    policy = workspace.model
    # if cfg.training.use_ema:
    #     policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
   
    cfg.task.env_runner.train_start_seed=10000
    cfg.task.env_runner.test_start_seed=5000
    cfg.task.env_runner.n_test=15
    cfg.task.env_runner.n_train=15
    cfg.task.env_runner.n_test_vis=15
    cfg.task.env_runner.n_train_vis=15
    cfg.task.env_runner.max_steps=500
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy,train_dataloader)
    
    # dump log to json
    json_log = dict()
    
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            wandb_run.log({"pusht": wandb.Video(value._path)})
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    wandb.save(f"{out_path}")

if __name__ == '__main__':
    main()
