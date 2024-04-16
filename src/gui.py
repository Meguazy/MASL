import tkinter as tk
import pandas as pd

class AgentMovementApp:

    last_position_dict = dict()

    def __init__(self, master, csv_file_brain, csv_file_periphery):
        self.master = master
        self.master.title("Agent Movement Visualization")

        self.canvas = tk.Canvas(self.master, width=400, height=900, bg="white")
        self.canvas.pack()

        self.canvas.create_rectangle(0, 401, 1600, 499, fill="gray")  # Create a line to separate brain and periphery agents
        
        self.agent_markers = {}  # Dictionary to store agent markers by agent ID
        self.agent_colors = {}  # Dictionary to store agent colors by agent ID

        self.brain_df = self.read_csv(csv_file_brain, 0)
        self.periphery_df = self.read_csv(csv_file_periphery, 1)
        self.create_agent_markers()  # Create agent markers for tick 1

    def read_csv(self, csv_file, env_id):
        df = pd.read_csv(csv_file, usecols=["tick", "agent_id", "agent_type", "x", "y"])
        df.sort_values(by=['tick', 'agent_id'], inplace=True)  # Sort dataframe by tick
        if env_id == 0:   
            self.brain_agents = df['agent_id'].unique()  # Get unique agent IDs
            self.brain_movements = df[['agent_id', 'x', 'y', 'agent_type']].values.tolist()  # Get agent movements
        elif env_id == 1:   
            self.periphery_agents = df['agent_id'].unique()  # Get unique agent IDs        
            self.periphery_movements = df[['agent_id', 'x', 'y', 'agent_type']].values.tolist()
        return df              

    def create_agent_markers(self):
        self.get_agent_colors()
        for agent_id, agent_type in self.agent_colors.items():
            marker = None
            if agent_type == "red":
                marker = self.canvas.create_oval(0, 0, 10, 10, fill=agent_type)  # Create marker for each agent ID with color based on agent type
            elif agent_type == "blue" or agent_type == "green" or agent_type == "yellow":
                marker = self.canvas.create_rectangle(0, 0, 10, 10, fill=agent_type)  # Create marker for each agent ID with color based on agent type
            elif agent_type == "#00ffff":
                marker = self.canvas.create_polygon(0, 0, 10, 0, 5, 10, fill=agent_type)
            self.agent_markers[str(agent_id)] = marker

    def get_agent_colors(self):
        # Assign different colors for each agent type
        colors = {"0": "red", "1": "yellow", "3": "green", "4": "blue", "5": "#00ffff"}  # Add more agent types and colors as needed
        for agent_id, x, y, agent_type in self.brain_movements:
            if agent_id not in self.agent_colors:
                self.agent_colors[agent_id] = colors[str(agent_type)]
        for agent_id, x, y, agent_type in self.periphery_movements:
            if agent_id not in self.agent_colors:
                self.agent_colors[agent_id] = colors[str(agent_type)]

    def animate_movement(self):
        for tick in range(1, 3):  # Loop through each tick
            tick_df_brain = None
            tick_df_periphery = None
            print(tick)
            if tick > 1:
                tick_df_brain = self.brain_df[self.brain_df['tick'] == tick]# and self.brain_df['agent_type'] == 1]
                tick_df_brain = tick_df_brain[tick_df_brain['agent_type'] == 1]
            else:
                tick_df_brain = self.brain_df[self.brain_df['tick'] == tick]
                tick_df_periphery = self.periphery_df[self.periphery_df['tick'] == tick]  # Get dataframe for each tick
           
            tick_df_brain.to_csv('output/brain_tick_' + str(i) + '.csv', index=False)

            if tick_df_brain is not None:
                tick_brain_movements = tick_df_brain[['agent_id', 'x', 'y', 'agent_type']].values.tolist()  # Get agent movements for each tick
                for agent_id, x, y, _ in tick_brain_movements:
                    marker = self.agent_markers[str(agent_id)]
                    #print(marker)
                    self.canvas.move(marker, x*10, y*10)  # Scale up for visualization
                    self.master.after(0, self.canvas.update())  # Update canvas after 1 second

            if tick_df_periphery is not None:
                tick_periphery_movements = tick_df_periphery[['agent_id', 'x', 'y', 'agent_type']].values.tolist()  # Get agent movements for each tick
                for agent_id, x, y, _ in tick_periphery_movements:
                    marker = self.agent_markers[str(agent_id)]                                        
                    self.canvas.move(marker, x*10, (y*10)+500)  # Scale up for visualization
                    self.master.after(0, self.canvas.update())  # Update canvas after 1 second
            

  
                    
        self.master.mainloop()

if __name__ == "__main__":
    for i in range(1,3):
        print("AÀaaaa")
        print(i)
    root = tk.Tk()
    app = AgentMovementApp(root, "./output/brain_set_2.csv", "./output/periphery_set.csv")
    app.animate_movement()
    