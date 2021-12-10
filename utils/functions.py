import random
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

from robomimic.config import config_factory
from robomimic.algo import algo_factory
from .dataset import SequenceDataset
from torch.utils.data import DataLoader

def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.
    Args:
        dataset_path (str): path to the dataset hdf5
    """
    dataset = SequenceDataset(
        hdf5_path = dataset_path,
        obs_keys = ( # observations we want to appear in batches
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ),
        dataset_keys = ( # can optionally specify more keys here if they should appear in batches
            "actions",
            "rewards",
            "dones",
        ),
        load_next_obs = True,
        frame_stack = 1,
        seq_length = 1, # length-10 temporal sequences
        pad_frame_stack = True,
        pad_seq_length = True, # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask = False,
        goal_mode = None,
        hdf5_cache_mode = "all", # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr = True,
        hdf5_normalize_obs = False,
        filter_by_attribute = None, # can optionally provide a filter key here
    )

    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset = dataset,
        sampler = None, # no custom sampling logic (uniform sampling)
        batch_size = 1, # batches of size 100
        shuffle = True,
        num_workers = 0,
        drop_last = True # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader

def get_example_model(dataset_path, device):
    """
    Use a default config to construct a BC model.
    """

    # default BC config
    config = config_factory(algo_name="bc")

    # read config to set up metadata for observation types (e.g. detecting image observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # read dataset to get some metadata for constructing model
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_modalities=sorted((
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        )),
    )

    # make BC model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        modality_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    return model

def index_split(l, split, seed = 0):
    '''
        Function to randomly select indices
    '''
    random.seed = seed
    idx = list(range(l))
    random.shuffle(idx)
    return idx[0:split], idx[split::]

# write a function to convert a ob_dict to a numpy array and a dict
def obdict2array(obs, obs_key):
    idx_dict = {}
    ob_array = []
    cur = 0
    for key in obs.keys():
        if key in obs_key:
            idx_dict[key] = (cur, cur + len(obs[key]))
            cur += len(obs[key])
            ob_array.append(obs[key])
    return np.concatenate(ob_array), idx_dict

# function to handle acitons
def actions2array(act_ensemble):
    return np.stack(act_ensemble)

# function to handle a list
def obdicts2array(ob_ac_log, obs_key):
    ob_array = []
    ac_array = []
    for ob_dict in ob_ac_log:
        for obs, ac in ob_dict:
            tmp_ob_array, ob_idx_dict = obdict2array(obs.copy(), obs_key)
            tmp_ac_array = actions2array(ac.copy())
            ob_array.append(tmp_ob_array)
            ac_array.append(tmp_ac_array)
    return ob_idx_dict, np.stack(ob_array), np.stack(ac_array)

# first process the rest data to save as a dictionary
def dataset_obs(dataset, idx, obs_keys):
    obs_dict = {}
    for i in idx:
        tmp_obs_array = []
        for key in obs_keys:
            tmp_obs_array.append(dataset[i]['obs'][key].reshape(-1))
        obs_dict[i] = np.concatenate(tmp_obs_array)
    return obs_dict

def UpdateTrain(train_idx, test_idx, closest_idx_list):
    for ele in closest_idx_list:
        if ele in test_idx:
            test_idx.remove(ele)
        if ele not in train_idx:
            train_idx.append(ele)

def FindClosestIdx(left_obs_dict, tmp_obs, selected, k):
    import heapq
    heap_list = []
    for _ in range(k):
        heapq.heappush(heap_list, (-999999, -1))
    for idx in left_obs_dict.keys():
        dist = ObsDist(tmp_obs, left_obs_dict[idx])
        if dist < -heap_list[0][0]:
            if idx not in selected:
                heapq.heappushpop(heap_list, (-dist, idx))
    return heap_list

def ObsDist(tmp_obs, left_obs):
    return np.linalg.norm(tmp_obs - left_obs, 2)

def random_index_select(left_idx, num):
    random.shuffle(left_idx)
    selected_idx = left_idx[0:num]
    return selected_idx

# define a function to select idx to add
def error_idx_selection(prediction_error, target_number, train_idx, error_type):
    selected_idx = []
    assert error_type in ('max', 'min',)
    reverse = True if error_type == 'max' else False

    # error without normalizing
    error_rank = []
    for error_idx in range(len(prediction_error)):
        error_rank.append((error_idx, np.linalg.norm(prediction_error[error_idx])))

    # rank according to error and var
    error_rank.sort(key = lambda x:x[1], reverse=reverse)
    error_idx_list = [ele[0] for ele in error_rank]
    idx = 0

    while len(selected_idx) < target_number and idx >= 0 and\
    len(train_idx) + len(selected_idx) < len(error_idx_list):
        if error_idx_list[idx] not in train_idx:
            selected_idx.append(error_idx_list[idx])
        idx += 1

    return selected_idx