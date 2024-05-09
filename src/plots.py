import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data: pd.DataFrame):
    data.plot()
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("./counts/NOlevo_NOcarbo.csv", usecols=["num_deads", "num_alphas"])
    plot_data(data)