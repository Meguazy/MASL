import tkinter as tk
import pandas as pd

class AgentMovementApp:
    def __init__(self, master, csv_file):
        self.master = master
        self.master.title("Agent Movement Visualization")

        self.canvas = tk.Canvas(self.master, width=500, height=500, bg="white")
        self.canvas.pack()

        self.agent_markers = {}  # Dictionary to store agent markers by agent ID

        self.read_csv(csv_file)

    def read_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        self.agent_types = df['agent_type'].unique()  # Get unique agent types
        self.movements = df[['agent_id', 'x', 'y', 'agent_type']].values.tolist()

    def create_agent_markers(self):
        colors = self.get_agent_colors()
        for agent_type in self.agent_types:
            marker_color = colors[agent_type]
            marker = self.canvas.create_oval(0, 0, 10, 10, fill=marker_color)  # Create marker for each agent type
            self.agent_markers[agent_type] = marker

    def animate_movement(self):
        self.create_agent_markers()
        for agent_id, x, y, agent_type in self.movements:
            marker = self.agent_markers[agent_type]
            self.canvas.move(marker, x * 50, y * 50)  # Scale up for visualization
            self.master.after(1000, self.canvas.update())  # Update canvas after 1 second
        self.master.mainloop()

    def get_agent_colors(self):
        # Assign different colors for each agent type
        colors = {"001": "red", "002": "green", "003": "blue"}  # Add more agent types and colors as needed
        return colors

if __name__ == "__main__":
    root = tk.Tk()
    app = AgentMovementApp(root, "agent_movement_multiple_types.csv")
    app.animate_movement()
