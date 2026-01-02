from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from algorithms import atari_agents
from utils.utils import Logger


def make_env(env_name: str, seed: int, resize: int = 84) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        if len(env.observation_space.shape):  # pixel obs
            env = gym.wrappers.ResizeObservation(env, (resize, resize))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk


def make_save(cfg: DictConfig) -> str:
    time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y.%m.%d.%H.%M.%S"))
    save_path = Path(cfg.base_path) / time
    cfg.save_path = str(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if cfg.model:
        model_path = save_path / "models"
        cfg.model_path = str(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
    return time

def run(args: DictConfig, stdout) -> None:
    time = make_save(args)
    args.seed = random.randint(0, 100000)
    OmegaConf.save(args, Path(args.save_path) / "config.yaml")
    sys.stdout = Logger(stdout, str(Path(args.save_path) / "logs.txt"))
    print("============================================================")
    print("saving at:", args.save_path)
    # create train env and eval env
    envs = gym.vector.SyncVectorEnv(
        [make_env("ALE/" + args.env, args.seed + i, args.resize) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    eval_env = gym.vector.SyncVectorEnv(
        [make_env("ALE/" + args.env, args.seed, args.resize)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda_deterministic

    # create agent
    agent = atari_agents[args.algo.algo_name](args, envs, eval_env, device)
    avg_reward, std_reward = agent.run()
    print("============================================================")
    print("saving at:", time, "avg reward:", avg_reward, std_reward)
    print("============================================================")
    sys.stdout.close()


@hydra.main(config_path="cfgs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    stdout = sys.stdout
    for r in range(cfg.run):
        run(cfg, stdout)


if __name__ == "__main__":
    main()
