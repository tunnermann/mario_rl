import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_file, output_dir):
    # Ler o CSV
    data = pd.read_csv(csv_file)

    # Plotar recompensas médias
    plt.figure(figsize=(10, 6))
    plt.plot(data['timesteps'], data['reward_mean'], label='Recompensa Média')
    plt.xlabel('Timesteps')
    plt.ylabel('Recompensa Média')
    plt.title('Recompensa Média durante o Treinamento')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/reward_plot.png")
    plt.close()

    # Plotar perda média
    plt.figure(figsize=(10, 6))
    plt.plot(data['timesteps'], data['loss_mean'], label='Perda Média', color='orange')
    plt.xlabel('Timesteps')
    plt.ylabel('Perda Média')
    plt.title('Perda Média durante o Treinamento')
    plt.legend()
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.close()

if __name__ == "__main__":
    csv_file = "dqn/metrics/dqn_metrics.csv"
    output_dir = "plots"
    plot_metrics(csv_file, output_dir)
    print(f"Gráficos salvos em {output_dir}")
