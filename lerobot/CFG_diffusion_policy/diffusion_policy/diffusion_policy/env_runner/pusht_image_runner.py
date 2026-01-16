import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
import cv2 
import random
import pickle
from diffusion_policy.common.pytorch_util import dict_apply 
from collections import deque
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
from itertools import cycle
from diffusion_policy.common.goal_utli import angle_between_vectors
from diffusion_policy.common.ood_util import find_threshold_by_percentile
class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=2,
            n_train_vis=3,
            train_start_seed=5000,
            n_test=2,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=8000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test
        
        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        
        # train
        n_train=15
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        n_test=15
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.past_obs=deque(maxlen=20)
        self.past_action_list=list()
        self.loss_list=list()
    def run(self, policy: BaseImagePolicy,vae):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # self.visualizer= PymunkKeypointManager.create_from_pusht_env(env, max_trail_length=200)
        # plan for rollout
        # import pdb;pdb.set_trace()
        # step= 0
        # i=0
        # for data in cycle(dataloader):
        #     # input=dict_apply(data['obs'], lambda x: x[:,:self.n_obs_steps,...])
        #     # input['image'].to('cuda:0')
        #     # input['agent_pos'].to('cuda:0')
        #     # data['goal'].to('cuda:0')
        #     # data['pre'].to('cuda:0')
        #     policy.to('cpu')
        #     # loss=policy.kkd_detect(data)
        #     recon_loss=policy.compute_loss(data)['recon_loss']
        #     self.loss_list.append(recon_loss)
        #     print(recon_loss)
        #     i=i+1
        #     if i==2000:
        #         break
        # # import pdb;pdb.set_trace()
        # error=np.mean(self.loss_list)
        # # threshold=find_threshold_by_percentile(np.array(self.loss_list),percentile=99)
        # print('----')
        # print(error)
        #     # zeros_tensor=torch.zeros_like(data['goal']).to('cpu')
            # print(f"Model device: {next(vae.parameters()).device}")
            # print(f"Input image device: {input['image'].device}")
            # print(f"Goal device: {data['goal'].device}")
            # return_dict=vae(input,data['pre'],data['goal'])
            # return_dict1=vae(input,input['image'],input['image'])
            # return_dict2=vae(input,zeros_tensor,zeros_tensor)
            # recon_loss=return_dict['recon_loss']
            # goal=return_dict['key_outputs']
            # print(f'recon_loss{recon_loss}')
            # print(f'key_loss{return_dict['key_loss']}')
            # print('goal and recon are all obs')
            # print(f'key_loss{return_dict1['key_loss']}')
            # print(f'recon_loss{return_dict1['recon_loss']}')
            # print('goal and recon are all obs')
            # print(return_dict2['key_loss'])
            # print(return_dict2['recon_loss'])
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        action_list=list()
        action1_list=list()
        temp_angle=[[] for _ in range(n_envs)]
        self.past_step_action=list()
        error=list()
        # allocate data
        all_video_paths = [None] * n_envs
        all_rewards = [None] * n_envs
        with open('action_step.pkl', 'rb') as file:
            action_step=pickle.load(file)
        with open('action_prompt.pkl', 'rb') as file:
            action_prompt=pickle.load(file)
        # import pdb;pdb.set_trace()
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
                this_local_slice=slice(0,this_n_active_envs+n_diff)
                this_global_slice=slice(start,end+n_diff)
            assert len(this_init_fns) == n_envs
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # out = cv2.VideoWriter(f'{chunk_idx}.mp4', fourcc, 30.0, (512, 512))
            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])
            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            step_index=0
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
           
            done = False
            step=0
            angle=list()
            
            t=0
            while not done:
                # step=step+1
                # # create obs dict
                # self.past_obs.append(obs)
                # nobs = policy.normalizer.normalize(obs)
                # this_nobs = dict_apply(nobs, 
                # lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                # pre_img,key_img,_,_=vae(this_nobs)
                # recon_loss=vae.recon_loss(pre_img,self.past_obs[-1])
               
                
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    import pdb;pdb.set_trace()
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
               
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                
                recon_loss=vae.compute_loss(obs_dict)['recon_loss']
                error.append(recon_loss)
                print('-----')
                t=t+1
                print(f't is {t}')
                
                print(recon_loss)
                print('------')
                # recon_loss=0.0690
                # # print(recon_loss)
                # # recon_loss=0
                # # print(recon_loss)
                # # if recon_loss >=234500:
                # # if recon_loss>=0.06899:
                # #     import pdb;pdb.set_trace()
                # if recon_loss>=0.0689:
                #     print('abnormal')
                #     error_step=1
                #     if error_step>=len(self.past_action_list):  
                #         # error_step=len(self.past_action_list)
                #         break
                #     # my_prompt=torch.flip(self.past_action_list[-1],dims=[0]).cuda()
                #     my_prompt=self.past_action_list[-error_step]
                #     prompt=torch.zeros_like(my_prompt)
                #     prompt[12:16]=my_prompt[12:16]
                #     prompt[5]=my_prompt[5]
                #     my_prompt=prompt.cuda()
                #     original_step_action=self.past_step_action[-error_step]
                #     step_action=torch.flip(self.past_step_action[-error_step],dims=[1])
                #     # for i in range(step_action.shape[0]):
                #     #     if i not in [5,12,13,14,15]:
                #     #         for j in range(step_action.shape[1]):
                #     #             step_action[i][j]=step_action[i][0]
                #     # with torch.no_grad():
                #     #         action_dict=policy.predict_action(obs_dict,positive_prompt=my_prompt)
                #     #         np_action_dict = dict_apply(action_dict,
                #     #             lambda x: x.detach().to('cpu').numpy())
                #     # action = torch.flip(torch.from_numpy(np_action_dict['action']),dims=[0])
                   
                #     action=torch.cat([step_action,step_action],dim=1)
                #     obs, reward, done, info = env.step(action)
                #     done=np.all(done)
                #     # error_step=error_step+1
                   
                 
                #         # print(i)
                #         # print(self.past_action_list)
                #     # my_prompt=self.past_action_list[-1].cuda()

                #     # step_action=self.past_step_action[-1]
                    
                #     np_obs_dict = dict(obs)
                #     obs_dict = dict_apply(np_obs_dict, 
                #     lambda x: torch.from_numpy(x).to(
                #         device=device))
                #     recon_loss=vae.compute_loss(obs_dict)['recon_loss']
                #     print('*****')
                    
                #     print(recon_loss)
                #     print('*****')
                #     while(recon_loss>=0.0689):
                #         print('error again')
                #         error_step=error_step+1
                #         my_prompt=self.past_action_list[-error_step]
                #         prompt=torch.zeros_like(my_prompt)
                #         prompt[12:16]=my_prompt[12:16]
                #         prompt[5]=my_prompt[5]
                #         my_prompt=prompt.cuda()
                #         original_step_action=self.past_step_action[-error_step]
                #         step_action=torch.flip(self.past_step_action[-error_step],dims=[1])
                #         action=torch.cat([step_action,step_action],dim=1)
                #         obs, reward, done, info = env.step(action)
                #         done=np.all(done)
                #         np_obs_dict = dict(obs)
                #         obs_dict = dict_apply(np_obs_dict, 
                #     lambda x: torch.from_numpy(x).to(
                #         device=device))
                #         recon_loss=vae.compute_loss(obs_dict)['recon_loss']
                #         # print('*****')
                    
                #         # print(recon_loss)
                #         # print('*****')
                #         # error_step=error_step+1
                #         # if error_step==3:
                #         #     import pdb;pdb.set_trace()
                #         #     error_step=1
                #         # continue
                #     #     action=torch.cat([step_action,original_step_action],dim=1)
                #     #     obs, reward, done, info = env.step(action)
                #     #     done=np.all(done)
                #     with torch.no_grad():
                #         obs_dict0= dict_apply(obs_dict,
                #                 lambda x: x.detach()[5:6])
                #         obs_dict1 = dict_apply(obs_dict,
                #              lambda x: x.detach()[0:5])
                #         obs_dict2 = dict_apply(obs_dict,
                #              lambda x: x.detach()[6:12])
                #         obs_dict3 = dict_apply(obs_dict,
                #              lambda x: x.detach()[12:16])
                #         obs_dict4 = dict_apply(obs_dict,
                #              lambda x: x.detach()[16:])
                #         action_dict0=policy.predict_action(obs_dict0,negative_prompt=my_prompt[5:6])
                #         action_dict3=policy.predict_action(obs_dict3,negative_prompt=my_prompt[12:16])
                #         action_dict1=policy.predict_action(obs_dict1)
                #         action_dict2=policy.predict_action(obs_dict2)
                #         action_dict4=policy.predict_action(obs_dict4)
                #         for key in action_dict1.keys():
                #             action_dict1[key]=torch.cat([action_dict1[key],action_dict0[key],action_dict2[key],action_dict3[key],action_dict4[key]],dim=0)
                #         self.past_action_list.append(action_dict1['action_pred'])
                #         np_action_dict = dict_apply(action_dict1,
                #             lambda x: x.detach().to('cpu').numpy())
                #         action = torch.from_numpy(np_action_dict['action'])
                #         self.past_step_action.append(action)
                      
                #         for i in range(action.shape[0]):
                #             u1=np.array(action[i][-1]-step_action[i][0])
                #             u2=np.array(step_action[i][-1]-step_action[i][0])
                #             temp_angle[i].append(angle_between_vectors(u1,u2))
                #         action=torch.cat([action,original_step_action],dim=1)
                #         obs, reward, done, info = env.step(action)
                        
                #         done=np.all(done)
                #          # print(action.shape)
                       
                        
                        
                      

                # else:
                #     print('normal')
                    # print('normal_angle')
                    # recon_loss=vae.compute_loss(obs_dict)['recon_loss']
                    # print(recon_loss)
                    # print('normal_angle')
                with torch.no_grad():
                    action_dict=policy.predict_action(obs_dict)
                self.past_action_list.append(action_dict['action_pred'])
                    
                np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())
                    # past_action = np_action_dict['action']
                action = torch.from_numpy(np_action_dict['action'])
                print(action.shape)
                    # action_dict=policy.predict_action(obs_dict)
                    # np_action_dict = dict_apply(action_dict,
                    #     lambda x: x.detach().to('cpu').numpy())
                    # # past_action = np_action_dict['action']
                    # action1 = torch.from_numpy(np_action_dict['action'])
                self.past_step_action.append(action)
                action=torch.cat([action,action],dim=1)
                    
                obs, reward, done, info = env.step(action)
                    # print(obs)
                pbar.update(action.shape[1])
                done=np.all(done)
                    
                    
            
                    #     my_prompt=action_dict['action_pred']
                    #     action_list.append(action_dict['action'])
                    # action_dict=policy.predict_action(obs_dict)
                    
                    #     action_dict = policy.predict_action(obs_dict)
                    #     self.past_action.append(action_dict['pre_action'])
                    #     my_prompt=action_dict['action_pred']
                    #     action_list.append(action_dict['action'])
                    #     action_dict=policy.predict_action(obs_dict)
                
                # action_pre=action_step.pop(0)
                # my_prompt=action_prompt.pop(0)
                # device_transfer
               
                # import pdb;pdb.set_trace()
                # if step%1==0:
                #     goal=vae.recon_key(obs_dict)
                
                #     obs_dict['image']=goal['image']
                # with torch.no_grad():
                #     action_dict=policy.predict_action(obs_dict)
                    
                #     np_action_dict = dict_apply(action_dict,
                #     lambda x: x.detach().to('cpu').numpy())
                #     my_prompt=np_action_dict['action_pred']
                #     step_action = torch.from_numpy(np_action_dict['action'])
                
                #     action_dict=policy.predict_action(obs_dict,negative_prompt=my_prompt)
                # np_action_dict = dict_apply(action_dict,
                #     lambda x: x.detach().to('cpu').numpy())
                # action = torch.from_numpy(np_action_dict['action'])
                # prompt=np_action_dict['action_pred']
                # action_list.append(prompt)
                # action1_list.append(action)
                # step env
                # temp_angle=0
                # print(action.shape)
                # for i in range(action.shape[0]):
                #     u1=np.array(action[i][-1]-step_action[i][0])
                #     u2=np.array(step_action[i][-1]-step_action[i][0])
                #     temp_angle=temp_angle+angle_between_vectors(u1,u2)
                #     angle.append(temp_angle/action.shape[0])
                # action=torch.cat([action,step_action],dim=1)
                
                
                # obs, reward, done, info = env.step(action)
                
                   
               
                
                # done = np.all(done)
                # past_action = action
                
                # # frame = env.render(mode='rgb_array')
                # # frame = self.visualizer.draw_keypoints_pose(frame, pose_map)
                # # out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pbar.update(action.shape[1])
                # print(temp_angle)
            
            pbar.close()
            # with open('action_prompt.pkl', 'wb') as file:
            #     pickle.dump(action_list,file)
            # with open('action_step.pkl','wb') as file:
            #     pickle.dump(action1_list,file)
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
           
        # clear out video buffer
        _ = env.reset()
        print('final error')
        
        print(np.array(error).mean(axis=0))
        # log
        # print(angle)
        # angle=angle/step
        # print(f'angle{angle}')
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        # with open('action_prompt.pkl', 'wb') as file:
        #     pickle.dump(action_list,file)
        # with open('action_step.pkl','wb') as file:
        #     pickle.dump(action1_list,file)
        angle=np.array(temp_angle).mean(axis=1)

        for i in range(n_envs):
            print(angle[i])
            if i>=n_inits:
                seed = self.env_seeds[0]
                prefix = self.env_prefixs[0]
            else:
                seed = self.env_seeds[i]
                prefix = self.env_prefixs[i]
            
            max_reward = np.max(all_rewards[i])
            angle_temp=angle[i]
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}-{i}'] = max_reward
            log_data[prefix+f'angle-{i}'] = angle_temp.item()
            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}-{i}'] = sim_video
                # log_data[prefix+f'angle-{i}'] = angle_temp
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        return log_data
