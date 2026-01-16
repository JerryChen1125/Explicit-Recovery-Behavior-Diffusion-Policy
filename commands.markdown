```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=calibrate
```

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=teleoperate
```

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Test." \
    --control.repo_id=$USER/gello_test \
    --control.num_episodes=1 \
    --control.push_to_hub=false
```
"--robot.type=gello" ,"--control.type=record","--control.fps=30", "--control.single_task=\"Test.\"","--control.repo_id=$USER/gello_test","--control.num_episodes=1","--control.push_to_hub=false"

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=dataset_0316/pick_place_teapot \
    --control.episode=0
```
"--robot.type=gello" ,"--control.type=replay","--control.fps=30", "--control.repo_id=datasets_temp/gello_test","--control.episode=0"

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=dataset_0316/pick_place_teapot \
  --policy.type=diffusion \
  --output_dir=outputs/train/0316_diffusion_pick_place_teapot \
  --job_name=0316_diffusion_pick_place_teapot \
  --policy.device=cuda \
  --wandb.enable=true
```
"--dataset.repo_id=datasets_temp/gello_test", "--policy.type=act", "--output_dir=outputs/train/act_gello_test", "--job_name=act_gello_test", "--policy.device=cuda", "--wandb.enable=true"

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=dataset_0318/pick_place_teapot \
    --control.single_task="Pick and place teapot on the red cloth." \
    --control.num_episodes=30 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=20 \
    --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.resume=true
```

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=dataset_test/0 \
    --control.single_task="Test." \
    --control.num_episodes=30 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=20 \
    --control.reset_time_s=10 \
    --control.push_to_hub=false
```
"--robot.type=gello", "--control.type=record", "--control.fps 30", "--control.repo_id=dataset_test/0", "--control.single_task=\"Test.\"", "--control.num_episodes=30", "--control.warmup_time_s=2", "--control.episode_time_s=20", "--control.reset_time_s=10", "--control.push_to_hub=false"

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Pick and place teapot on the red cloth." \
    --control.repo_id=dataset_0316/eval_0316_diffusion_pick_place_teapot \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=200 \
    --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.policy.path=outputs/train/0316_diffusion_pick_place_teapot/checkpoints/last/pretrained_model
```

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Pick and place teapot on the red cloth." \
    --control.repo_id=dataset_0316/eval_0316_pick_place_teapot \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=1000000 \
    --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.policy.path=outputs/train/0316_pick_place_teapot/checkpoints/last/pretrained_model

```
"--robot.type=gello", "--control.type=record", "--control.fps=30", "--control.single_task=\"Test.\"", "--control.repo_id=datasets_temp/eval_gello_test", "--control.num_episodes=10", "--control.warmup_time_s=2", "--control.episode_time_s=30", "--control.reset_time_s=10", "--control.push_to_hub=false","--control.policy.path=outputs/train/act_gello_test/checkpoints/080000/pretrained_model"

