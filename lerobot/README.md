# 真机实验操作指南
* 更换gripper，固定相机、桌面、机械臂、光源位置，启动机械臂和光源电源，连接数据线和网线，确保急停按钮打开。
* 禁用网络代理，连接USB以太网至xarm1，打开ufactory-studio-client，enable机械臂，设置Motion/Parameters/Initial Position。（J1：0，J2：-15，J3：-55，J4：-180，J5：-60，J6：0）
* 打开realsense-viewer，检查相机是否正常运行。
* 命令行： conda activate lerobot
## 数据采集
每次采集开始前固定好摇臂起始位置，建议参照机械臂Initial Position，结束前回到起始位置。注意等待warmup和reset结束后log显示“Recording episode”再进行操作，否则相机不会记录。
* 代码：在control_robot.py中，注释### for deploying ###部分，解注释### for recording ###部分。
### 单人操作（一次采一条）
注意经常会发生相机卡死未记录等情况（结束时log不会立刻显示“Stop recording”），建议每次采集后手动检查outputs中image是否变化。若warmup时间过长，有时杀死进程重新跑有帮助。
* 命令行：python lerobot/scripts/control_robot.py --robot.type=gello --control.type=record --control.fps=30 --control.repo_id=lerobot/test --control.single_task="test" --control.root=outputs --control.num_episodes=1 --control.warmup_time_s=5 --control.episode_time_s=30 --control.push_to_hub=false
* 命令行参数：repo_id和single_task不重要但必须，root可以换成绝对路径（eg. /home/sealab/lerobot_outputs/task1/01），warmup_time_s如果设置太短可能出现前几帧相机未录制，episode_time_s根据任务难度可自由调整。
### 多人操作（一次采多条）
按“->”键结束采集当前episode或跳过reset，按“<-”键重新采集当前episode，按“Esc”键提前结束所有采集。注意一次采多条的文件结构和一次采一条不同，需另外实现脚本将data、image与episode解耦或修改预处理代码逻辑。
* 命令行：python lerobot/scripts/control_robot.py --robot.type=gello --control.type=record --control.fps=30 --control.repo_id=lerobot/test --control.single_task="test" --control.root=outputs --control.num_episodes=50 --control.warmup_time_s=5 --control.episode_time_s=30 --control.reset_time_s=10 --control.push_to_hub=false
* 命令行参数：repo_id和single_task不重要但必须，root可以换成绝对路径（eg. /home/sealab/lerobot_outputs/task1），num_episodes根据训练需求可自由调整，episode_time_s根据任务难度可自由调整，reset_time_s需确保将实验场景归位。
### 注意事项
遥操过程中尽量避免打到限位，如要调整，修改Safety/Safety Boundary（密码为admin）。解除限位时首先减小J2、J3，再归位至Initial Position。
## 数据预处理
基于一次采一条的文件结构，将所有数据合并成一个HDF5文件，并按需压缩image尺寸。
* 代码：在data盘下lerobot_outputs/merge.py中，修改input_path、output_path和resize参数。
## 数据重现
注意每次只能重现单条episode。
* 代码：在control_robot.py中，注释### for deploying ###部分，解注释### for recording ###部分。
* 命令行：python lerobot/scripts/control_robot.py --robot.type=gello --control.type=replay --control.fps=30 --control.repo_id=lerobot/test --control.root=outputs --control.episode=0
* 命令行参数：repo_id不重要但必须，root必须和采集路径一样，episode为索引值。
## 模型部署
按“->”键结束部署当前episode或跳过reset，按“Esc”键提前结束所有部署。
* 代码：在control_robot.py中，注释### for recording ###部分，解注释### for deploying ###部分，load_policy()的cfg.training.seed可随意设置。
* 命令行：python lerobot/scripts/control_robot.py --robot.type=gello --control.type=record --control.fps=30 --control.repo_id=lerobot/test --control.single_task="test" --control.num_episodes=20 --control.warmup_time_s=5 --control.episode_time_s=60 --control.reset_time_s=10 --control.push_to_hub=false
* 命令行参数：repo_id和single_task不重要但必须，episode_time_s根据任务难度合理设置（超时记为失败），reset_time_s需确保将实验场景和机械臂归位。
### Baseline 代码
* 在control_robot.py中，修改load_policy()的ckpt_path。
* 在control_robot.py和control_utils.py中，注释### for policy with error detector ###部分，解注释### for baseline policy ###部分。
### Error Detector 代码
* 在control_robot.py中，修改load_policy()的ckpt_path和vae_ckpt。
* 在control_robot.py和control_utils.py中，注释### for baseline policy ###部分，解注释### for policy with error detector ###部分。
### 注意事项
部署过程中机械臂可能会打到限位，如有必要，按下急停按钮。解除限位时首先减小J2、J3，再归位至Initial Position。