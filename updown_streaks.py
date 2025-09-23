import pandas as pd
from daily_returns_Algorithms import daily_returns
def streaks(returns: list[float | None]) -> dict:
    up = 0 # counts the current up streak
    down = 0 # counts the current down streak
    highest_up_streak = 0 # the longest up streak
    highest_down_streak = 0 # the longest down streak
    num_up_streaks = 0 # the number of up streaks started
    num_down_streaks = 0 # the number of down streaks started
    total_up_days = 0 # total number of up days
    total_down_days = 0 # total number of down days

    for r in returns:
        if r is None: # skip none values
            continue
        if r == 0: # if return is 0 reset both streaks
            if up > 0 and up > highest_up_streak: # checks the current up streak and if it is the highest
                highest_up_streak = max(highest_up_streak, up) # updates the highest up streak if current up streak is higher
                up = 0
            if down > 0 and down > highest_down_streak: # checks the current down streak and if it is the highest
                highest_down_streak = max(highest_down_streak, down) # updates the highest down streak if current down streak is higher
                down = 0
            continue

        elif r > 0:
            total_up_days += 1 # if return is pos, up streak has started and total of up days increases
            if down > 0 and down > highest_down_streak: # checks the current down streak and if it is the highest
                    highest_down_streak = max(highest_down_streak, down) # updates the highest down streak if current down streak is higher
            elif down > 0: # if the downstreak is not the highest, reset the down streak
                down = 0 
            up += 1 # since we are in a up streak increase the up streak count
            if up == 1: # if the up streak is 1 it means a new up streak has started
                num_up_streaks += 1
            highest_up_streak = max(highest_up_streak, up) # updates the highest up streak if current up streak is higher

        elif r < 0:
            total_down_days += 1 #since return is neg, down streak has started and total num of down streak increases
            if up > 0 and up > highest_up_streak: #sees if the current up streak is the highest, and if it is update the highest up streak
                highest_up_streak = max(highest_up_streak, up)
            elif up > 0: # if the up streak is not the highest set up streak counter back to 0
                up = 0
            down += 1 # since now we are in a down streak, the down streak count increases
            if down == 1: # if down streak = 1 means new down streak has started
                num_down_streaks += 1
            highest_down_streak = max(highest_down_streak, down) # updates the highest down streak if current down streak is higher 

    # Final check at the end of the loop to ensure the longest streaks are recorded
    if up > 0 and up > highest_up_streak:
        highest_up_streak = max(highest_up_streak, up)
    if down > 0 and down > highest_down_streak:
        highest_down_streak = max(highest_down_streak, down)

    return{
        "Highest up streak": highest_up_streak,
        "Highest down streak": highest_down_streak,
        "Number of up streaks": num_up_streaks,
        "Number of down streaks": num_down_streaks,
        "Total number of up days": total_up_days,
        "Total number of down days": total_down_days
    }

if __name__ == "__main__":
    try:
        # Load dataset
        df = pd.read_csv("Clean_data/AAPL_stock.csv", parse_dates=["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

        # Extract close prices
        prices = df["Close"].astype(float).tolist()

        # Calculates the daily returns using the daily_returns function from daily_returns_Algorithms file
        returns = daily_returns(prices)

        # Retuns a dictionary containing the streaks stats from the consecutive up and down days
        # Analyses the up and down streaks from the daily returns file
        streak = streaks(returns)

        print("\n===== Apple Stock Up/Down Streaks Analysis =====")
        print("-----------------------------------------------")
        print(f"Number of up streaks: {streak['Number of up streaks']}")
        print(f"Number of down streaks: {streak['Number of down streaks']}")
        print(f"Highest up streak: {streak['Highest up streak']} days")
        print(f"Highest down streak: {streak['Highest down streak']} days")
        print(f"Total number of up days: {streak['Total number of up days']}")
        print(f"Total number of down days: {streak['Total number of down days']}")
        print("===============================================\n")

    except FileNotFoundError:
        print("Could not find Clean_data/AAPL_stock.csv — check your path.")