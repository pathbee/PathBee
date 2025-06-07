import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file with proper column names, skipping the header row
col_names = [
    'v (vertex)', 'w (vertex)', 'dist (hops)', 'query_time (microseconds)',
    'v_out_index_num (count)', 'w_in_index_num (count)'
]
df = pd.read_csv('results/road-usroads/dc_query.csv', sep='\t', names=col_names, skiprows=1)

# Convert the relevant columns to numeric, coercing errors
for col in ['v_out_index_num (count)', 'w_in_index_num (count)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN in these columns
df = df.dropna(subset=['v_out_index_num (count)', 'w_in_index_num (count)'])

# Debug: print first few rows and dtypes
def debug_df(df):
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDtypes:")
    print(df.dtypes)
    print(f"Number of rows: {len(df)}")

debug_df(df)

# Calculate the sum of v_out_index_num and w_in_index_num
df['total_index_count'] = df['v_out_index_num (count)'] + df['w_in_index_num (count)']

# Create the distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='total_index_count', bins=30, kde=True)

# Customize the plot
plt.title('Distribution of Total Index Count (v_out + w_in)')
plt.xlabel('Total Index Count (number of labels)')
plt.ylabel('Frequency (count)')

# Add mean and median lines
mean_value = df['total_index_count'].mean()
median_value = df['total_index_count'].median()
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f} labels')
plt.axvline(median_value, color='green', linestyle='--', label=f'Median: {median_value:.2f} labels')
plt.legend()

# Save the plot
plt.savefig('index_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics
print(f"\nStatistics for Total Index Count:")
print(f"Mean: {mean_value:.2f} labels")
print(f"Median: {median_value:.2f} labels")
print(f"Min: {df['total_index_count'].min()} labels")
print(f"Max: {df['total_index_count'].max()} labels")
print(f"Standard Deviation: {df['total_index_count'].std():.2f} labels")
