import tkinter as tk
import pandas as pd
import loguru

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
        self.neurons_status_df = pd.read_csv("./output/neuron_status.csv")
        self.microglia_status_df = pd.read_csv("./output/microglia_status.csv")
        self.astrocytes_status_df = pd.read_csv("./output/astrocyte_status.csv")
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
        # print(self.agent_colors)
        for agent_id, value in self.agent_colors.items():
            marker = None
            agent_type = value[1]
            color = value[0]

            if agent_type == 0:
                marker = self.canvas.create_oval(0, 0, 10, 10, fill=color)  # Create marker for each agent ID with color based on agent type
            elif agent_type == 3:
                marker = self.canvas.create_rectangle(0, 0, 10, 10, fill=color)
            elif agent_type == 4:
                marker = self.canvas.create_polygon(0, 0, 10, 0, 5, 10, fill=color)
            else:
                marker = self.canvas.create_oval(0, 0, 10, 10, fill=color)
            self.agent_markers[agent_id] = marker

    def get_agent_colors(self):
        # Assign different colors for each agent type
        colors = {0: "red", 1: "yellow", 3: "red", 4: "red", 5: "#00ffff", 6: "#6646e2", 7: "#1be8e4"}  # Add more agent types and colors as needed
        for agent_id, x, y, agent_type in self.brain_movements:
            if agent_id not in self.agent_colors:
                self.agent_colors[agent_id] = [colors[agent_type], agent_type]
        for agent_id, x, y, agent_type in self.periphery_movements:
            if agent_id not in self.agent_colors:
                self.agent_colors[agent_id] = [colors[agent_type], agent_type]

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
            loguru.logger.debug(f"Starting processing tick {tick}")

            self.update_tick_display(tick)
            if tick > 1:
                delay = 500  # Delay in milliseconds between ticks

            brain_tick_df = find_tick_diff(self.brain_df[self.brain_df['tick'] == tick - 1], self.brain_df[self.brain_df['tick'] == tick])
            brain_tick_df_dropped = brain_tick_df.drop(brain_tick_df[brain_tick_df['status'] == 'NOT_CHANGED'].index)
            brain_tick_df_dropped = brain_tick_df_dropped.reset_index()

            loguru.logger.debug(f"Starting brain movement processing for tick {tick}")

            for _, row in brain_tick_df_dropped.iterrows():
                status = row["status"]
                marker = self.agent_markers[row["agent_id"]]
                
                if(status == "NEW"):
                    self.canvas.move(marker, row["x_updated"]*10, row["y_updated"]*10)
                elif(status == "MOVED"):
                    self.canvas.move(marker, (row["x_updated"] - row["x_initial"])*10, (row["y_updated"] - row["y_initial"])*10)
                elif(status == "REMOVED"):
                    self.canvas.move(marker, -row["x_initial"]*10, -row["y_initial"]*10)
            
            loguru.logger.debug(f"Finished brain movement processing for tick {tick}")
            loguru.logger.debug(f"Starting brain status processing for tick {tick}")

            tick_status_brain_df = self.neurons_status_df[self.neurons_status_df['tick'] == tick]
            tick_status_brain_df.reset_index()

            for _, row in tick_status_brain_df.iterrows():
                marker = self.agent_markers[row["agent_id"]]
                if row["is_alive"]:
                    if row["is_alpha"]:
                        self.canvas.itemconfig(marker, fill="green")
                    else:
                        self.canvas.itemconfig(marker, fill="red")
                else:
                    self.canvas.itemconfig(marker, fill="black")

            loguru.logger.debug(f"Finished brain status processing for tick {tick}")
            loguru.logger.debug(f"Starting microglia status processing for tick {tick}")

            tick_status_microglia_df = self.microglia_status_df[self.microglia_status_df['tick'] == tick]
            tick_status_microglia_df.reset_index()

            for _, row in tick_status_microglia_df.iterrows():
                marker = self.agent_markers[row["agent_id"]]
                if row["is_activated"]:
                    self.canvas.itemconfig(marker, fill="black")
                else:
                    self.canvas.itemconfig(marker, fill="red")

            loguru.logger.debug(f"Finished microglia status processing for tick {tick}")
            loguru.logger.debug(f"Starting astrocytes status processing for tick {tick}")

            tick_status_astrocytes_df = self.astrocytes_status_df[self.astrocytes_status_df['tick'] == tick]
            tick_status_astrocytes_df.reset_index()

            for _, row in tick_status_astrocytes_df.iterrows():
                marker = self.agent_markers[row["agent_id"]]
                if row["is_activated"]:
                    self.canvas.itemconfig(marker, fill="black")
                else:
                    self.canvas.itemconfig(marker, fill="red")

            loguru.logger.debug(f"Finished astrocytes status processing for tick {tick}")
            loguru.logger.debug(f"Starting periphery movement processing for tick {tick}")

            periphery_tick_df = find_tick_diff(self.periphery_df[self.periphery_df['tick'] == tick - 1], self.periphery_df[self.periphery_df['tick'] == tick])
            periphery_tick_df_dropped = periphery_tick_df.drop(periphery_tick_df[periphery_tick_df['status'] == 'NOT_CHANGED'].index)
            periphery_tick_df_dropped = periphery_tick_df_dropped.reset_index()
            if tick == 35:
                periphery_tick_df_dropped.to_csv("periphery_tick_df.csv")

            for _, row in periphery_tick_df_dropped.iterrows():
                status = row["status"]
                marker = self.agent_markers[row["agent_id"]]
                if(status == "NEW"):
                    loguru.logger.debug(f"Creating new marker for agent {row['agent_id']}")
                    self.canvas.move(marker, row["x_updated"]*10, (row["y_updated"]*10)+500)
                elif(status == "MOVED"):
                    self.canvas.move(marker, (row["x_updated"] - row["x_initial"])*10, ((row["y_updated"] - row["y_initial"])*10)+500)
                elif(status == "REMOVED"):
                    self.canvas.delete(marker)
            
            loguru.logger.debug(f"Finished periphery movement processing for tick {tick}")

            self.master.after(delay, self.canvas.update())

            loguru.logger.debug(f"Finished processing tick {tick}")
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
    