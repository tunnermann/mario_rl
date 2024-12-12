import pandas as pd

reward_log = pd.read_csv("model_a2c3/reward_log.csv", index_col='timesteps')
plot = reward_log.plot()
plot.figure.savefig('./model_a2c3/reward_plot.png')

