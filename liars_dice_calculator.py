from scipy.stats import binom
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def dice_roll(num_dice=15): 
    """
    Simulate rolling a number of dice and return the results.
    
    Parameters:
    num_dice (int): Number of dice to roll
    
    Returns:
    list: Results of the dice rolls
    """
    return [random.randint(1, 6) for _ in range(num_dice)]




def liars_dice_calc(D=15, p=1/6, c=0, k=0):
    """
    Calculate the probability of at least c successes in D trials
    with success probability p.
    
    Parameters:
    D (int): Number of trials
    p (float): Probability of success on each trial
    c (int): Minimum number of successes required
    k (int): Number of successes already observed
    
    Returns:
    float: Probability of at least c successes
    """
    c_required = c - k
    if c_required <= 0:
        return 1.0  # Already have enough successes
    prob = 1 - binom.cdf(c_required - 1, D, p)
    return prob

def print_odds_table(D=15, p=1/6):
    """
    Print a table of probabilities for different values of c.
    """
    print(f"{'c':<10}{'Probability (%)':<20}")
    print("-" * 30)
    for c in range(0, D+1):
        prob = liars_dice_calc(D=D, p=p, c=c)
        print(f"{c:<10}{prob * 100:<20.10f}")


def simulate_dice_rolls(num_rolls=1000, num_dice=15):
    """
    Simulate rolling dice multiple times and return the results.
    
    Parameters:
    num_rolls (int): Number of times to roll the dice
    num_dice (int): Number of dice to roll each time
    
    Returns:
    list: Results of the dice rolls
    """
    return [dice_roll(num_dice) for _ in range(num_rolls)]

def analyize_rolls(rolls):
    """
    Anaalyze the results of the dice rolls. We want to capture the percentage for each round
    that each number appeared.

    We also want to capture the percentage over the rounds that all 6 numbers appeared.
    """
    counts = {i: 0 for i in range(1, 7)}
    number_present_in_round = {i: 0 for i in range(1, 7)}
    total_rolls = len(rolls)
    die_per_roll = len(rolls[0])


    for roll in rolls:
        temp_appearance = {i: 0 for i in range(1, 7)}
        for die in roll:
            counts[die] += 1
            if not temp_appearance[die]:
                temp_appearance[die] += 1
        for i in range(1, 7):
            if temp_appearance[i]:
                number_present_in_round[i] += 1
    

    print("Percentage of rounds where each number appeared:")
    for k, v in number_present_in_round.items():
        print(f"Number {k}: {v / total_rolls * 100:.2f}%")
    

    all_numbers_percentage = sum(1 for round in rolls if len(set(round)) == 6) / total_rolls * 100

    percentages = {k: (v / (total_rolls * die_per_roll)) * 100 for k, v in counts.items()}


    # print("Percentage of each number appearing:")
    # for number, percentage in percentages.items():
    #     print(f"Number {number}: {percentage:.2f}%")
    print(f"Percentage of rounds where all numbers appeared: {all_numbers_percentage:.6f}%")
 

def simulate_and_analyze_frequencies(num_simulations=10000, num_dice=20):
    """
    Simulate dice rolls and analyze the frequency distribution.
    
    Parameters:
    num_simulations (int): Number of simulation rounds
    num_dice (int): Number of dice rolled in each simulation
    
    Returns:
    pd.DataFrame: DataFrame containing frequency counts for each dice value
    """
    # Initialize a list to store the counts of each number in each simulation
    frequency_data = []
    
    # Run simulations
    for _ in range(num_simulations):
        roll = dice_roll(num_dice)
        counts = {i: roll.count(i) for i in range(1, 7)}
        frequency_data.append(counts)
    
    # Convert to DataFrame
    df = pd.DataFrame(frequency_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # First subplot: Box plot
    sns.boxplot(data=pd.melt(df), x='variable', y='value', ax=ax1)
    ax1.set_title(f'Distribution of Dice Counts Over {num_simulations} Simulations')
    ax1.set_xlabel('Dice Value')
    ax1.set_ylabel('Count')
    
    # Second subplot: Average distribution
    # Calculate average percentage for each occurrence count across all dice values
    avg_distributions = []
    for possible_count in range(num_dice + 1):  # 0 to num_dice
        percentages = []
        for dice_value in range(1, 7):
            count = (df[dice_value] == possible_count).sum()
            percentage = (count / num_simulations) * 100
            percentages.append(percentage)
        avg_percentage = sum(percentages) / 6
        if avg_percentage > 0:  # Only include non-zero percentages
            avg_distributions.append((possible_count, avg_percentage))
    
    # Unzip the data for plotting
    counts, percentages = zip(*avg_distributions)
    
    # Create the average distribution plot
    ax2.bar(counts, percentages, alpha=0.7, color='blue', label='Data')
    ax2.set_title('Average Distribution of Dice Counts')
    ax2.set_xlabel('Number of Occurrences')
    ax2.set_ylabel('Average Percentage (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(counts, percentages, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(counts), max(counts), 100)
    ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.8, label='Trend Line')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Calculate and print frequency of specific counts
    print("\nFrequency of specific counts:")
    for dice_value in range(1, 7):
        value_counts = df[dice_value].value_counts().sort_index()
        print(f"\nDice value {dice_value}:")
        for count, frequency in value_counts.items():
            percentage = (frequency / num_simulations) * 100
            print(f"{count} occurrences: {frequency} times ({percentage:.2f}%)")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the probability of at least c successes in D trials.')
    parser.add_argument('-D', dest='D', type=int, default=15, help='Number of dice (default: 15)')
    parser.add_argument('-p', '--probability', dest='p', type=float, default=1/6, help='Probability of success on each trial (default: 1/6)')
    parser.add_argument('-c', '--called', dest='c',  type=int, default=0, help='Number of dice called (default: 0)')
    parser.add_argument('-k', '--num-in-hand', dest='k', type=int, default=0, help='Number of called dice you have in hand (default: 0)')
    parser.add_argument('-s', '--simulate', action='store_true', help='Run frequency simulation')

    args = parser.parse_args()
    print(args)
    
    if args.simulate:
        df = simulate_and_analyze_frequencies()
    else:
        print_odds_table(args.D, args.p)
        probability = liars_dice_calc(args.D, args.p, args.c, args.k)
        print(f"Probability of call being truth: {probability * 100:.10f}%")