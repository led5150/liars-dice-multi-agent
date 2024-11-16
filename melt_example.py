import pandas as pd

# Original "wide" format with an extra identifier column
df = pd.DataFrame({
    'player': ['Alice', 'Bob', 'Charlie'],
    'round': [1, 1, 1],
    1: [3, 2, 1],
    2: [2, 4, 2],
    3: [4, 1, 3]
})

print("Original DataFrame:")
print(df)
print("\nShape:", df.shape)

# Melt while keeping 'player' and 'round' as identifier columns
melted_df = pd.melt(df, 
                    id_vars=['player', 'round'],
                    var_name='dice_value', 
                    value_name='count')

print("\nMelted DataFrame:")
print(melted_df)
print("\nShape:", melted_df.shape) 