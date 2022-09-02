# CDPRgym
 Cable Driven Parallel Robot gym model


For RL zoo: 
clone this repo into the RL-zoo folder from 
```
https://github.com/DLR-RM/rl-baselines3-zoo
```

Add the following into hyperparams/ppo.yml

```
# custom
CDPR-v0:
  vec_env_wrapper: 
    - stable_baselines3.common.vec_env.VecMonitor
    - stable_baselines3.common.vec_env.vec_normalize.VecNormalize
  n_envs: 20
  n_timesteps: !!float 1e7
  policy: 'MlpPolicy'
  n_steps: 20_000
  gae_lambda: 0.95
  gamma: 0.9
  n_epochs: 10
  ent_coef: 0.0
  learning_rate: !!float 1e-3
  clip_range: 0.2
  use_sde: True
  sde_sample_freq: 4
  ```
Also add the following to utils/import_envs.py
```
register(id="CDPR-v0", entry_point="from CDPRGYM.envs.cdpr import CDPRenv")

```

try running 

```
python train.py --algo ppo --env "CDPR-v0" --vec-env "subproc"
```
