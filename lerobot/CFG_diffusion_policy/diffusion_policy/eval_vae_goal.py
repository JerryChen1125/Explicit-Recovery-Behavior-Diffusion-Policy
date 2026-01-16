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
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-c', '--checkpoint_vae', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, checkpoint_vae,output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
            dir=str(output_dir),
         
           
            settings=wandb.Settings(start_method='fork')
        )
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    vae_payload = torch.load(open(checkpoint_vae, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg_vae=vae_payload['cfg']
    cfg.training.seed=42
   
    cls = hydra.utils.get_class(cfg._target_)
    cls_vae = hydra.utils.get_class(cfg_vae._target_)
    workspace_vae= cls_vae(cfg_vae, output_dir=output_dir)
    workspace_vae: BaseWorkspace
    workspace_vae.load_payload(vae_payload, exclude_keys=None, include_keys=None)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    original_random=workspace.original_state
    policy = workspace.model
    # vae=workspace_vae.model
    vae=workspace_vae.model
    device = torch.device(device)
    policy.to(device)
    # vae.to(device)
    # vae.eval()
    policy.eval()
    cfg.n_action_steps=4
    cfg.task.env_runner.train_start_seed=8500
    cfg.task.env_runner.test_start_seed=5000
    cfg.task.env_runner.n_test=15
    cfg.task.env_runner.n_train=15
    cfg.task.env_runner.n_test_vis=15
    cfg.task.env_runner.n_train_vis=15
    cfg.task.env_runner.max_steps=1000
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy,vae)
    
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
