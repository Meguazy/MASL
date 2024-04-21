import tkinter as tk
import pandas as pd

class AgentMovementApp:

    last_position_dict = dict()

    def __init__(self, master, csv_file_brain, csv_file_periphery):
        self.master = master
        self.master.title("Agent Movement Visualization")

        self.canvas = tk.Canvas(self.master, width=500, height=900, bg="white")
        self.canvas.pack()

        self.canvas.create_rectangle(0, 401, 900, 499, fill="gray")  # Create a line to separate brain and periphery agents
        self.canvas.create_rectangle(400, 0, 420, 900, fill="black")  # Create a line to separate brain and periphery agents

        self.create_tick_display()  # Create a rectangle to display the current tick number
        
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
            elif agent_type == "blue" or agent_type == "green" or agent_type == "yellow" or agent_type == "#6646e2":
                marker = self.canvas.create_rectangle(0, 0, 10, 10, fill=agent_type)  # Create marker for each agent ID with color based on agent type
            elif agent_type == "#00ffff":
                marker = self.canvas.create_polygon(0, 0, 10, 0, 5, 10, fill=agent_type)
            self.agent_markers[agent_id] = marker

    def get_agent_colors(self):
        # Assign different colors for each agent type
        colors = {"0": "red", "1": "yellow", "3": "green", "4": "blue", "5": "#00ffff", "6": "#6646e2"}  # Add more agent types and colors as needed
        for agent_id, x, y, agent_type in self.brain_movements:
            if agent_id not in self.agent_colors:
                self.agent_colors[agent_id] = colors[str(agent_type)]
        for agent_id, x, y, agent_type in self.periphery_movements:
            if agent_id not in self.agent_colors:
                self.agent_colors[agent_id] = colors[str(agent_type)]

    def create_tick_display(self):
        # Create a rectangle to display the current tick number
        self.tick_display = self.canvas.create_rectangle(430, 10, 480, 40, fill="lightgray")
        self.tick_text = self.canvas.create_text(455, 25, text=f"Tick: {1}")

    def update_tick_display(self, tick):
        # Update the tick display with the new tick number
        self.canvas.itemconfig(self.tick_text, text=f"Tick: {tick}")

    def animate_movement(self):
        delay = 0
        max_tick = self.brain_df['tick'].max()
        for tick in range(1, max_tick):
            self.update_tick_display(tick)
            if tick > 1:
                delay = 300
            tick_previous_df = self.brain_df[self.brain_df['tick'] == tick - 1]
            tick_current_df = self.brain_df[self.brain_df['tick'] == tick]
            brain_tick_df = find_tick_diff(tick_previous_df, tick_current_df)
            brain_tick_df_dropped = brain_tick_df.drop(brain_tick_df[brain_tick_df['status'] == 'NOT_CHANGED'].index)
            brain_tick_df_dropped = brain_tick_df_dropped.reset_index()

            for _, row in brain_tick_df_dropped.iterrows():
                status = row["status"]
                marker = self.agent_markers[row["agent_id"]]
                print(marker)
                
                if(status == "NEW"):
                    self.canvas.move(marker, row["x_updated"]*10, row["y_updated"]*10)
                elif(status == "MOVED"):
                    self.canvas.move(marker, (row["x_updated"] - row["x_initial"])*10, (row["y_updated"] - row["y_initial"])*10)
                elif(status == "REMOVED"):
                    self.canvas.delete(marker)
                
                
            if tick == 1:
                tick_previous_df = self.periphery_df[self.periphery_df['tick'] == tick - 1]
                tick_current_df = self.periphery_df[self.periphery_df['tick'] == tick]
                periphery_tick_df = find_tick_diff(tick_previous_df, tick_current_df)

                for _, row in periphery_tick_df.iterrows():
                    status = row["status"]
                    marker = self.agent_markers[row["agent_id"]]
                    if(status == "NEW"):
                        self.canvas.move(marker, row["x_updated"]*10, (row["y_updated"]*10)+500)
                    elif(status == "MOVED"):
                        self.canvas.move(marker, (row["x_updated"] - row["x_initial"])*10, ((row["y_updated"] - row["y_initial"])*10)+500)
                    elif(status == "REMOVED"):
                        self.canvas.delete(marker)
                        
            self.master.after(delay, self.canvas.update())
        self.master.mainloop()
        

def find_tick_diff(tick_previous_df, tick_current_df):
    merged_df = pd.merge(tick_previous_df, tick_current_df, on='agent_id', how="outer", suffixes=('_initial', '_updated'))

    def determine_status(row):
        if pd.isnull(row['x_updated']) and pd.isnull(row['y_updated']):
            return 'REMOVED'
        elif pd.isnull(row['x_initial']) and pd.isnull(row['y_initial']):
            return 'NEW'
        elif row['x_initial'] == row['x_updated'] and row['y_initial'] == row['y_updated']:
            return 'NOT_CHANGED'
        else:
            return 'MOVED'

    merged_df['status'] = merged_df.apply(determine_status, axis=1)

    merged_df.drop(['tick_initial', 'tick_updated'], axis=1, inplace=True)
    
    return merged_df

if __name__ == "__main__":
    root = tk.Tk()
    app = AgentMovementApp(root, "./output/brain_set.csv", "./output/periphery_set.csv")
    app.animate_movement()
    