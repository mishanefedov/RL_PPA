import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data from CSV
csv_file = 'results.csv'
pdf_file = 'success_rate_plot.pdf'
data = pd.read_csv(csv_file)

# Convert the necessary columns to numeric data types
data['Success Rate'] = pd.to_numeric(data['Success Rate'], errors='coerce')
data['Mean Total Reward'] = pd.to_numeric(data['Mean Total Reward'], errors='coerce')
data['Mean Steps'] = pd.to_numeric(data['Mean Steps'], errors='coerce')

# Filter out the summary row and any non-task-specific data if needed
data = data[~data['Title'].str.contains("all tasks", case=False, na=False)]

# Group by 'Title' and calculate mean if there are multiple entries per title
grouped_data = data.groupby('Title').mean()

# Plotting Success Rate
with PdfPages(pdf_file) as pdf:
    plt.figure(figsize=(10, 6))
    grouped_data['Success Rate'].plot(kind='bar', color='teal')
    # grouped_data['Mean Steps'].plot(kind='bar', color='teal')
    plt.title('Success Rate by Title')
    plt.ylabel('Success Rate')
    # plt.ylabel('Mean Steps')
    plt.xlabel('Title')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"Success rate plot saved to {pdf_file}")
