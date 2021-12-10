import numpy as np
import os, time, json, imageio
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.obs_utils as ObsUtils

from collections import OrderedDict
from robomimic.envs.env_base import EnvBase

class RolloutPolicyEnsemble(object):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    """
    def __init__(self, policy, obs_normalization_stats=None):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts
            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation modality keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.policy.set_eval()
        self.policy.reset()

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.
        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
        """
        if self.obs_normalization_stats is not None:
            ob = ObsUtils.normalize_obs(ob, obs_normalization_stats=self.obs_normalization_stats)
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        return ob

    def __repr__(self):
        """Pretty print network description"""
        return self.policy.__repr__()

    def __call__(self, ob, goal=None):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.
        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
            goal (dict): goal observation
        """
        ob = self._prepare_observation(ob)
        if goal is not None:
            goal = self._prepare_observation(goal)
        ac = self.policy.get_action(obs_dict=ob, goal_dict=goal)
        return TensorUtils.to_numpy(ac[0])

def run_rollout(
        policy,
        env,
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
    ):
    """
    Runs a rollout in an environment with the current network parameters.
    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.
        env (EnvBase instance): environment to use for rollouts.
        horizon (int): maximum number of steps to roll the agent out for
        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env
        render (bool): if True, render the rollout to the screen
        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip
        video_skip (int): how often to write video frame
        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered
    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(env, EnvBase)

    for sub_policy in policy:
        sub_policy.start_episode()

    ob_dict = env.reset()

    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    success = { k: False for k in env.is_success() } # success metrics

    OB_AC_log = []

    try:
        for step_i in range(horizon):

            # get action from policy
            ac_tmp = []
            for tmp_policy in policy:
                tmp_ac = tmp_policy(ob=ob_dict, goal=goal_dict)
                ac_tmp.append(tmp_ac)
            ac = sum(ac_tmp) / len(ac_tmp)

            OB_AC_log.append((ob_dict, ac_tmp.copy()))

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = env.render(mode="rgb_array", height=512, width=512)
                    video_writer.append_data(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results, OB_AC_log

def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.
    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).
    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.
        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.
        horizon (int): maximum number of steps to roll the agent out for
        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env
        num_episodes (int): number of rollout episodes per environment
        render (bool): if True, render the rollout to the screen
        video_dir (str): if not None, dump rollout videos to this directory (one per environment)
        video_path (str): if not None, dump a single rollout video for all environments
        epoch (int): epoch number (used for video naming)
        video_skip (int): how often to write video frame
        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered
        verbose (bool): if True, print results of each rollout

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts
        video_paths (dict): path to rollout videos for each environment
    """
    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        OB_AC_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info, OB_AC_log = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            OB_AC_logs.append(OB_AC_log)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths, OB_AC_logs
