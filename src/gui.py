import tkinter as tk
import pandas as pd

class AgentMovementApp:
    def __init__(self, master, csv_file):
        self.master = master
        self.master.title("Agent Movement Visualization")

        self.canvas = tk.Canvas(self.master, width=800, height=800, bg="white")
        self.canvas.pack()

        self.agent_markers = {}  # Dictionary to store agent markers by agent ID
        self.agent_colors = {}  # Dictionary to store agent colors by agent ID

        self.read_csv(csv_file)
        self.create_agent_markers()  # Create agent markers for tick 1

    def read_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.loc[df["tick"] == 1]
        print(df.count)
        df.sort_values(by='tick', inplace=True)  # Sort dataframe by tick
        self.agents = df['agent_id'].unique()  # Get unique agent IDs
        self.movements = df[['agent_id', 'x', 'y', 'agent_type']].values.tolist()

    def create_agent_markers(self):
        self.get_agent_colors()
        for agent_id, agent_type in self.agent_colors.items():
            print(agent_type)
            marker = None
            if agent_type == "red":
                marker = self.canvas.create_oval(0, 0, 10, 10, fill=agent_type)  # Create marker for each agent ID with color based on agent type
            elif agent_type == "blue" or agent_type == "green":
                marker = self.canvas.create_rectangle(0, 0, 10, 10, fill=agent_type)  # Create marker for each agent ID with color based on agent type
            self.agent_markers[agent_id] = marker

    def get_agent_colors(self):
        # Assign different colors for each agent type
        colors = {"0": "red", "3": "green", "4": "blue"}  # Add more agent types and colors as needed
        for agent_id, x, y, agent_type in self.movements:
            if agent_id not in self.agent_colors:
                self.agent_colors[agent_id] = colors[str(agent_type)]

    def animate_movement(self):
        for agent_id, x, y, _ in self.movements:
            marker = self.agent_markers[agent_id]
            print(marker)
            self.canvas.move(marker, x*20, y*20)  # Scale up for visualization
            self.master.after(0, self.canvas.update())  # Update canvas after 1 second
        self.master.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = AgentMovementApp(root, "./output/agent_counts_1.csv")
    app.animate_movement()
