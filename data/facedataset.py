import numpy as np
import os, random

from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, root_dir, data_names, phase='test', transform=None, verbose=False):
        self.transform = transform
        self.data_names = {}
        
        if verbose:
            print('--------------------------------------------------')
            print(f'{"build test dataset":20} - number of video')
            print('--------------------------------------------------')

        for num, data_name in enumerate(data_names):
            self.data_names[data_name] = num

            path_attack = os.path.join(root_dir,data_name, phase,'attack')
            path_live = path_attack.replace('attack','live')

            video_attack_list = [os.path.join(path_attack,video) for video in os.listdir(path_attack)]
            video_live_list = [os.path.join(path_live,video) for video in os.listdir(path_live)]
            try:
                self.video_list += video_attack_list + video_live_list
            except:
                self.video_list = video_attack_list + video_live_list

            if verbose:
                print(f'{path_attack.replace(root_dir,""):20} : {len(video_attack_list)}')
                print(f'{path_live.replace(root_dir,""):20} : {len(video_live_list)}')

    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        spoofing_label = int('live' in video_name)
        domain_label = self.data_names[video_name.split('/')[1]]
        image_x = self.sample_image(video_name)
        image_x = self.transform(image_x)

        sample = {
                    "image_x"   : image_x,
                    "label"     : spoofing_label,
                    "domain"    : domain_label,
                    "name"      : video_name,
                }
        
        return sample

    def sample_image(self, image_dir):
        frames = os.listdir(image_dir)
        frames_total = len(frames)
        image_id = np.random.randint(0, frames_total)
        image_path = os.path.join(image_dir,frames[image_id])

        return Image.open(image_path) 
    
class BalanceFaceDataset(Dataset):
    def __init__(self, root_dir, data_names, phase='train', transform=None, max_iter=4000, verbose=False):
        self.transform = transform
        self.max_iter = max_iter

        self.video_list = {}
        self.data_names = {}
        print(data_names, verbose)
        # set video dictionary of domain-wise and class-wise
        for num, data_name in enumerate(data_names):
            self.data_names[data_name] = num

            path = os.path.join(root_dir,data_name, phase, 'attack')
            video_list = [os.path.join(path,video) for video in os.listdir(path)]
            random.shuffle(video_list)
            self.video_list[path] = video_list

            path = path.replace('attack','live')
            video_list = [os.path.join(path,video) for video in os.listdir(path)]
            random.shuffle(video_list)
            self.video_list[path] = video_list
        
        if verbose:
            print('--------------------------------------------------')
            print(f'{"build train dataset":20} - number of video')
            print('--------------------------------------------------')
            for key in self.video_list:
                print(f'{key.replace(root_dir,""):20} : {len(self.video_list[key])}')

    def __len__(self):
        return self.max_iter
    
    def __getitem__(self, idx):
        sample = {}
        for video_key in self.video_list:
            try:
                video_name = next(self.video_list[video_key])
            except:
                video_list = [os.path.join(video_key, video) for video in os.listdir(video_key)]
                random.shuffle(video_list)
                self.video_list[video_key] = iter(video_list)
                video_name = next(self.video_list[video_key])

            spoofing_label = int('live' in video_name)
            domain_label = self.data_names[video_name.split('/')[1]]

            image_x = self.sample_image(video_name)
            image_x_view1 = self.transform(image_x)
            image_x_view2 = self.transform(image_x)

            key = f"{video_key.split('/')[-3]}-{video_key.split('/')[-1]}"
            sample[key] = {
                            "image_x_v1": image_x_view1,
                            "image_x_v2": image_x_view2,
                            "label"     : spoofing_label,
                            "domain"    : domain_label,
                            "name"      : video_name,
                        }
        return sample

    def sample_image(self, image_dir):
        frames = os.listdir(image_dir)
        frames_total = len(frames)
        if frames_total==0:
            print(image_dir)
        image_id = np.random.randint(0, frames_total)
        image_path = os.path.join(image_dir,frames[image_id])

        return Image.open(image_path) 