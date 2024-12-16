import pandas as pd

reward_log = pd.read_csv("model_ppo_multi_level/reward_log.csv", index_col='timesteps')
plot = reward_log.plot()
plot.figure.savefig('./model_ppo_multi_level/reward_plot.png')

