import numpy as np 
import json
from petrel_client.client import Client
import h5py 
from tqdm import tqdm
import io

def process_log(path, ceph_path=None, save_path=None):
    with open(path, 'r') as f:
        log_lines = [line.strip() for line in f]

    meta_info_list = [f for f in log_lines if "meta_info.h5" in f]
    del meta_info_list[-1] # xxx/meta_info.h5
    episode_id_list = [f.split('/')[-2] for f in meta_info_list]
    for episode_id in episode_id_list:
        assert len(episode_id) == 6
    episode_id_list.sort()

    conf_path = '~/petreloss.conf'
    client = Client(conf_path)
    json_save = []
    for idx, episode_id in enumerate(tqdm(episode_id_list)):
        meta_info_path = f'{ceph_path}/episodes/{episode_id}/meta_info.h5'
        meta_info_bytes = client.get(meta_info_path, enable_cache=True)
        meta_info_h5_file = h5py.File(io.BytesIO(meta_info_bytes))
        json_save.append([episode_id, int(meta_info_h5_file['length'][()])])
        

    with open(save_path, "w") as f:
        json.dump(json_save, f, indent=1)


def search(dataset_name, src_path, tgt_path):
    with open(src_path,  'r') as f:
        src_data = json.load(f)

    with open(tgt_path,  'r') as f:
        tgt_data = json.load(f)

    episode_list = [f[0] for f in tgt_data]

    for k, v in src_data.items():
        if dataset_name in k:
            # print(v["episode_idx"])
            if v["episode_idx"] in episode_list:
                print("v: ", v)
                break


def process_num_steps(log_path, json_path):
    with open(log_path, 'r') as f:
        log_lines = [line.strip() for line in f]
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    conf_path = '~/petreloss.conf'
    client = Client(conf_path)
    meta_info_list = [f for f in log_lines if "meta_info.h5" in f]
    del meta_info_list[-1] # xxx/meta_info.h5
    episode_id_list = [f.split('/')[-2] for f in meta_info_list]
    set_trace()
    for idx, episode_id in tqdm(enumerate(episode_id_list)):
        meta_info_path = f'{ceph_path}/episodes/{episode_id}/steps/{str(json_data[idx][1]-1).zfill(4)}/other.h5'
        # if idx > 100:
        #     break
        try:
            meta_info_bytes = client.get(meta_info_path, enable_cache=True)
            meta_info_h5_file = h5py.File(io.BytesIO(meta_info_bytes))
            # print("success")
        except:
            print("episode_id :", episode_id)

   #  episodes = ["088258", "089892", "090798", "090999", "091139"]


    
    


if __name__ == "__main__":
    dataset_name = f"droid_failure"
    path = f"/mnt/petrelfs/tianyang/Code/CoRL_Manipulation/gr1/data_info/{dataset_name}.log"
    log_file = f"/mnt/petrelfs/tianyang/Code/CoRL_Manipulation/gr1/data_info/{dataset_name}_numstep.log"
    save_path = f"/mnt/petrelfs/tianyang/Code/CoRL_Manipulation/gr1/data_info/{dataset_name}.json"
    ceph_path = f"s3://real_data/{dataset_name}"
    # process_log(path, ceph_path, save_path)

    # search(dataset_name="TRI", 
    #        src_path="/mnt/petrelfs/tianyang/Code/CoRL_Manipulation/gr1/data_info/droid_depth_episode_info.json", 
    #        tgt_path="/mnt/petrelfs/tianyang/Code/CoRL_Manipulation/gr1/data_info/droid_failure.json")

    process_num_steps(log_path=path,
                      json_path=save_path
                        )

# srun -p mozi-S1 --quotatype=spot python data_info/read_log.py | tee  /mnt/petrelfs/tianyang/Code/CoRL_Manipulation/gr1/data_info/droid_failure_numstep.log





