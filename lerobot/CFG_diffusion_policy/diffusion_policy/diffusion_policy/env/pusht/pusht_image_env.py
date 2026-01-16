from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv
import numpy as np
import cv2
import torch
from typing import Optional
class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
        self.coord_list=list()
        self.prompt_list=list()
    def _get_obs(self,prompt= None):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if prompt is not None:
            prompt=np.array(prompt)
            coord = (prompt / 512 * 96).astype(np.int32)
            
            
            self.prompt_list.append(coord)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(100,100,255), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
            for i in range(len(self.prompt_list)):
                coord=self.prompt_list[i]
                cv2.circle(img, coord, 1, (0, 255, 0), -1)
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            
            coord = (action / 512 * 96).astype(np.int32)
            
            
            self.coord_list.append(coord)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
            for i in range(len(self.coord_list)):
                coord=self.coord_list[i]
                cv2.circle(img, coord, 1, (0, 0, 255), -1)
                
        #         cv2.putText(img, f"{i}", coord+{}, 
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.01, (255, 255, 255), 2)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache
