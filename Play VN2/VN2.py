import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd

INDEX = ["Store", "Product"]
DATE_order_0 = '2024-04-08' # Last week before the start of the competition.
HOLDING_COST = 0.2
SHORTAGE_COST = 1.0
os.makedirs("Data/", exist_ok=True)


def generate_starting_position():
    """
    Create the starting position of all players.
    """
    
    sales = pd.read_csv(f"Data/Week 0 - {DATE_order_0} - Sales.csv").set_index(INDEX)

    state = pd.DataFrame(index=sales.index)
    
    state["Start Inventory"] = 0
    state["Sales"] = 0
    state["Missed Sales"] = 0
    state["End Inventory"] = np.ceil(sales.iloc[:,-52:].mean(axis=1)).astype(int)
    assert (state["End Inventory"] > 0).all()
    state["In Transit W+1"] = 0 #End inventory is good enough.
    assert (state["In Transit W+1"] >= 0).all()
    state["In Transit W+2"] = (np.maximum(0,
                                         (sales.iloc[:,-52:].mean(axis=1)*2 - state["End Inventory"] - state["In Transit W+1"]))
                               .round().astype(int))
    assert (state["In Transit W+2"] >= 0).all()
    state["Holding Cost"] = 0 #Holding cost per Store x Product
    state["Shortage Cost"] = 0 #Shortage cost per Store x Product
    state["Cumulative Holding Cost"] = 0
    state["Cumulative Shortage Cost"] = 0

    state.to_csv(f"Data/Week 0 - {DATE_order_0} - Initial State.csv")


def player_setup(player="Nicolas"):
    """
    Initialize a player by setting up the initial state (from a common file)
    """
    os.makedirs(f"Data/{player}", exist_ok=True)
    state = pd.read_csv(f"Data/Week 0 - {DATE_order_0} - Initial State.csv")
    state.to_csv(f"Data/{player}/Week 0 - {DATE_order_0} - State.csv", index=False)
    

def compute_dates(order):
    date = (pd.to_datetime(DATE_order_0) + pd.Timedelta(weeks=order)).strftime('%Y-%m-%d')
    previous_date = (pd.to_datetime(date) + pd.Timedelta(weeks=-1)).strftime('%Y-%m-%d')
    return date, previous_date


def generate_dummy_order(player, order):
    """
    Generates a dummy order at the end of order X, so we know the demand and state of order X.
    """
    date, previous_date  = compute_dates(order)
    sales = pd.read_csv(f"Data/Week {order-1} - {previous_date} - Sales.csv").set_index(INDEX)
    submission = sales.iloc[:,-10:].mean(axis=1).squeeze().round(0).astype(int)
    submission.rename("Orders")
    submission.to_csv(f"Data/{player}/Week {order} - {date} - order.csv", index=True)
    print_submission(order, submission)


def generate_zero_order(player, order):
    """
    Generates a zero order at the end of order X
    Required for the last two rounds.
    """
    date, previous_date  = compute_dates(order)
    sales = pd.read_csv(f"Data/Week {order-1} - {previous_date} - Sales.csv").set_index(INDEX)
    submission = (pd.DataFrame(index=sales.index, data=[[0]])
                  .squeeze()
                  .astype(int)
                  .rename("Orders"))
    submission.to_csv(f"Data/{player}/Week {order} - {date} - Order.csv", index=True)
    print_submission(order, submission)


def generate_benchmark_order(player, order):

    # Step 1 - Get Data
    date, previous_date  = compute_dates(order)
    in_stock = pd.read_csv("Data/Week 0 - In Stock.csv").set_index(INDEX)    
    sales = pd.read_csv(f"Data/Week {order-1} - {previous_date} - Sales.csv").set_index(INDEX)
    state = pd.read_csv(f"Data/{player}/Week {order-1} - {previous_date} - State.csv").set_index(INDEX)  
    sales.columns = pd.to_datetime(sales.columns)
    in_stock.columns = pd.to_datetime(in_stock.columns)
    sales[~in_stock] = np.nan #These are shortages, we'll put missing data

    # Step 2 - Make a Seasonal Moving Average Forecast

    # Step 2a - Compute Seasonal Factors

    # We compute *simple* multiplicative weekly seasonal parameters
    season = sales.mean().rename("Demand").to_frame()
    season["Week Number"] = season.index.isocalendar().week
    season = season.groupby("Week Number").mean() #Seasonal parameters (multiplicative) per week
    season = season / season.mean() #Normalize to one.

    # Step 2b - Un-seasonalize Demand

    sales_weeks = sales.columns.isocalendar().week
    sales_no_season = sales / (season.loc[sales_weeks.values]).values.reshape(-1)

    # Step 2c we make a forecast using a 13 weeks moving average (the number is arbitrary)
    base_forecast = sales_no_season.iloc[:,-13:].mean(axis=1) # That's the unseasonalized moving average of the last 8 weeks
    # We need a forecast for 3 weeks.
    f_periods = pd.date_range(start=sales.columns[-1], periods=10, inclusive="neither", freq="W-MON")
    forecast = pd.DataFrame(data=base_forecast.values.reshape(-1,1).repeat(len(f_periods), axis=1), 
                            columns=f_periods,
                            index=sales.index)
    # We need to seasonalize this for future forecast. 
    forecast = forecast * (season.loc[f_periods.isocalendar().week.values]).values.reshape(-1)

    # Step 3 use a forecast-driven order-up-to policy with 4 weeks as coverage.

    order_up_to = forecast.iloc[:,:4].sum(axis=1)
    net_inventory = state[["In Transit W+1", "In Transit W+2", "End Inventory"]].sum(axis=1)
    submission = (order_up_to - net_inventory).clip(lower=0).round(0).astype(int)
    submission.rename("Orders")
    
    submission.to_csv(f"Data/{player}/Week {order} - {date} - order.csv", index=True)
    print_submission(order, submission)

def print_submission(order, submission):
    print(f"Week {order} Start\tOrder #{order}:\t{submission.sum()}")


def extract_order(sales, player, order):
    """
    Extract and validate a order file.
    When we create the state of order X, we load the order from order X-1
    The file should, 
    - Contain three columns ('Store', 'Product', and a third one containing your order)
    - Have an index exactly similar to sales.index (in the same order)
    - Only contains positive round integer (no decimals)
    """
    
    date, previous_date  = compute_dates(order)
    # Try both Order.csv and order.csv for case sensitivity
    try:
        order_file = pd.read_csv(f"Data/{player}/Week {order} - {date} - Order.csv")
    except FileNotFoundError:
        order_file = pd.read_csv(f"Data/{player}/Week {order} - {date} - order.csv")
    order = order_file

    assert all([col in order. columns for col in INDEX]), "We miss columns 'Store' and/or 'Product' in your order"
    order = order.set_index(INDEX)
    assert (order.shape == (sales.shape[0], 1)), f"Your order doesn't include the correct number of rows (expected: {sales.shape[0]}, got: {order.shape[0]}) and/or columns (expected: 1, got: {order.shape[1]})"
    order.columns = ['Orders']
    assert (order.index == sales.index).all(), "The index [Store x Product] of your order doesn't match sales.index"
    assert not order.isna()['Orders'].any(), "Your order includes missing values."
    assert (order["Orders"].round(0).astype(int) == order["Orders"]).all(), "Your order includes decimals; it should only be round integers."
    assert (order["Orders"] >= 0).all(), "Your order contains negative orders"
    order["Orders"].round(0).astype(int) == order["Orders"]
    
    return order
    

def update_inventory_state(player="Nicolas", order=1): 
    """
    Main loops that update the inventory state based on the date.
    sales: pivot-like dataframe with *all* sales values - don't share it with players
    . state: dataframe with the inventory state of the player at the end of a given week
    """
    assert type(order) == int
    assert order >= 0 

    sales = pd.read_csv("Data/Not For Share - Sales - All.csv").set_index(INDEX) #Contains all the sales data.
    date, previous_date  = compute_dates(order)
    demand = sales[date]
    
    orders = extract_order(sales, player=player, order=order)
    previous_state = pd.read_csv(f"Data/{player}/Week {order-1} - {previous_date} - State.csv").set_index(INDEX)   

    state = pd.DataFrame(index=sales.index)
    state["Start Inventory"] = previous_state["End Inventory"] + previous_state["In Transit W+1"]
    state["Sales"] = state["Start Inventory"].clip(upper=demand)
    state["Missed Sales"] = demand - state["Sales"]
    state["End Inventory"] = state["Start Inventory"] - state["Sales"]
    state["In Transit W+1"] = previous_state["In Transit W+2"]
    state["In Transit W+2"] = orders
    state["Holding Cost"] = state["End Inventory"]*HOLDING_COST 
    state["Shortage Cost"] = state["Missed Sales"]*SHORTAGE_COST 
    state["Cumulative Holding Cost"] = previous_state["Cumulative Holding Cost"] + state["Holding Cost"]
    state["Cumulative Shortage Cost"] = previous_state["Cumulative Shortage Cost"] + state["Shortage Cost"]

    state.to_csv(f"Data/{player}/Week {order} - {date} - State.csv") #State at the end of the week.

    round_cost = state[["Holding Cost","Shortage Cost"]].sum().sum()    
    cumulative_cost = state[["Cumulative Holding Cost","Cumulative Shortage Cost"]].sum().sum()

    print(f"Week {order} End\t\tRound:\t{round_cost.round(1)}\tCumulative: {cumulative_cost.round(1)}")

    return round_cost, cumulative_cost
   
    
def main():
    """
    Dummy function to generate an example of playthrough.
    Just a dummy example. In practice, orders are played at the same time for all players based on the official calendar.
    """
    generate_starting_position()
    
    # Dummy playthrough
    player = "Nicolas"    
    player_setup(player) #Load the starting inventory state, it's the same for all players.

    print(player)
    for order in range(1, 9):
        if order <= 6:
            generate_dummy_order(player, order)
        else:
            generate_zero_order(player, order)
        round_cost, cumulative_cost = update_inventory_state(player, order)
        
    # Benchmark playthrough

    player = "Benchmark"    
    player_setup(player) #Load the starting inventory state, it's the same for all players.

    print("\n", player)
    for order in range(1, 9):
        if order <= 6:
            generate_benchmark_order(player, order)
        else:
            generate_zero_order(player, order)
        round_cost, cumulative_cost = update_inventory_state(player, order)