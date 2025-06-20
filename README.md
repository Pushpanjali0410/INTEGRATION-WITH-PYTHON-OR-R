# INTEGRATION-WITH-PYTHON-OR-R

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: S PUSHPANJALI

*INTERN ID*: CT06DF2186

*DOMAIN*: Power BI

*DURATION*: 6 WEEKS

*MENTOR* : NEELA SANTHOSH

## OUTPUT
![Screenshot 2025-06-17 091114](https://github.com/user-attachments/assets/8eef6629-2afe-4a84-9c9a-0eb11af05e33)

## DESCRIPTION

###### 1) Open PowerBI and select the blank report . 

###### 2) Click on import data from excel and select financials dataset . 

###### 3) Select the python visuals from visual pane .

###### 4) Add the below code :

```python
# Power BI Python Visual - 9-Visual Dashboard (Matplotlib Only)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get data from Power BI
df = dataset.copy()

# Debug: Show available columns
print("Available columns:", df.columns.tolist())

# Set up the visualization with a 3x3 grid
plt.style.use('ggplot')  # Using built-in ggplot style instead of seaborn
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 7

# Create figure with constrained layout
fig = plt.figure(figsize=(20, 20), constrained_layout=True)
gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)

# Create all 9 subplots
ax1 = fig.add_subplot(gs[0, 0])  # Butterfly chart
ax2 = fig.add_subplot(gs[0, 1])  # Box plot
ax3 = fig.add_subplot(gs[0, 2])  # Pie chart
ax4 = fig.add_subplot(gs[1, 0])  # Scatter plot
ax5 = fig.add_subplot(gs[1, 1])  # Line chart
ax6 = fig.add_subplot(gs[1, 2])  # Bar chart
ax7 = fig.add_subplot(gs[2, 0])  # Bubble chart
ax8 = fig.add_subplot(gs[2, 1])  # Waterfall
ax9 = fig.add_subplot(gs[2, 2])  # Correlation

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
fig.suptitle('Financial Dashboard (Matplotlib Only)', y=1.02, fontsize=12)

# Get numeric and categorical columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

## 1. Butterfly Chart (Profit vs COGS by Product)
if 'Profit' in df.columns and 'COGS' in df.columns and 'Product' in df.columns:
    try:
        top_products = df.groupby('Product')[['Profit', 'COGS']].mean().nlargest(5, 'Profit')
        y_pos = np.arange(len(top_products))
        
        ax1.barh(y_pos, top_products['Profit'], color='limegreen', label='Profit', height=0.4)
        ax1.barh(y_pos, -top_products['COGS'], color='tomato', label='COGS', height=0.4)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_products.index)
        ax1.set_title('Profit vs COGS (Butterfly)')
        ax1.legend(fontsize=6)
        ax1.axvline(0, color='black', linewidth=0.5)
    except Exception as e:
        ax1.text(0.5, 0.5, 'Error in butterfly chart', ha='center', va='center', fontsize=6)

## 2. Box Plot (Profit Distribution)
if 'Profit' in df.columns:
    try:
        df['Profit'].plot(kind='box', ax=ax2, patch_artist=True,
                         boxprops=dict(facecolor='lightgreen'))
        ax2.set_title('Profit Distribution')
    except:
        ax2.text(0.5, 0.5, 'No profit data', ha='center', va='center', fontsize=6)

## 3. Pie Chart (Discount Band Distribution)
if 'Discount Band' in df.columns:
    try:
        discount_counts = df['Discount Band'].value_counts()
        discount_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax3,
                           colors=['gold','lightcoral','lightgreen'],
                           wedgeprops=dict(width=0.3), startangle=90)
        ax3.set_ylabel('')
        ax3.set_title('Discount Band Distribution')
    except:
        ax3.text(0.5, 0.5, 'No discount data', ha='center', va='center', fontsize=6)

## 4. Scatter Plot (Units Sold vs Manufacturing Price)
if 'Units Sold' in df.columns and 'Manufacturing Price' in df.columns:
    try:
        ax4.scatter(df['Manufacturing Price'], df['Units Sold'], alpha=0.5, s=20)
        
        # Add regression line manually
        x = df['Manufacturing Price']
        y = df['Units Sold']
        m, b = np.polyfit(x, y, 1)
        ax4.plot(x, m*x + b, color='red')
        
        ax4.set_title('Units Sold vs Manufacturing Price')
    except:
        ax4.text(0.5, 0.5, 'Missing required data', ha='center', va='center', fontsize=6)

## 5. Line Chart (First Numeric Column Over Index)
if len(numeric_cols) > 0:
    try:
        df[numeric_cols[0]].plot(ax=ax5, color='purple', linewidth=1)
        ax5.set_title(f'{numeric_cols[0]} Trend')
        ax5.grid(True, alpha=0.3)
    except:
        ax5.text(0.5, 0.5, 'No numeric data', ha='center', va='center', fontsize=6)

## 6. Bar Chart (Product Distribution)
if 'Product' in df.columns:
    try:
        df['Product'].value_counts().head(5).plot(kind='bar', ax=ax6, color='orange')
        ax6.set_title('Top 5 Products')
        ax6.tick_params(axis='x', rotation=45)
    except:
        ax6.text(0.5, 0.5, 'No product data', ha='center', va='center', fontsize=6)
        
## 7. Product Profitability Matrix
if 'Profit' in df.columns and 'Product' in df.columns:
    try:
        product_stats = df.groupby('Product')['Profit'].agg(
            Avg_Profit='mean',
            Total_Profit='sum',
            Count='count'
        ).reset_index()
        
        max_profit = product_stats['Total_Profit'].abs().max()
        bubble_scale = 500
        bubble_sizes = (product_stats['Total_Profit'].abs()/max_profit) * bubble_scale
        
        scatter = ax7.scatter(
            x=range(len(product_stats)),
            y=product_stats['Avg_Profit'],
            s=bubble_sizes,
            c=np.where(product_stats['Avg_Profit'] >= 0, 'green', 'red'),
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        ax7.set_xticks(range(len(product_stats)))
        ax7.set_xticklabels(product_stats['Product'], rotation=45, ha='right')
        ax7.set_ylabel('Average Profit')
        ax7.set_title('Product Profitability')
        ax7.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax7.grid(True, alpha=0.3)
        
        # Add simplified legend
        ax7.scatter([], [], c='green', s=50, label='Profitable')
        ax7.scatter([], [], c='red', s=50, label='Loss')
        ax7.legend(fontsize=6)
    except Exception as e:
        ax7.text(0.5, 0.5, 'Error showing profitability', ha='center', va='center', fontsize=6)

## 8. Waterfall Chart (Profit by Country)
if 'Profit' in df.columns and 'Country' in df.columns:
    try:
        profit_by_country = df.groupby('Country')['Profit'].sum()
        top_countries = profit_by_country.abs().nlargest(5).index
        profit_data = profit_by_country[top_countries].sort_values()
        
        cumulative = profit_data.cumsum()
        pos = profit_data.where(profit_data > 0, 0)
        neg = profit_data.where(profit_data < 0, 0)
        
        ax8.bar(profit_data.index, pos, color='green', label='Profit', width=0.6)
        ax8.bar(profit_data.index, neg, color='red', label='Loss', width=0.6)
        ax8.plot(profit_data.index, cumulative, 'bo-', markersize=4, label='Cumulative')
        
        ax8.set_title('Profit/Loss by Country')
        ax8.legend(fontsize=6)
        ax8.tick_params(axis='x', rotation=45)
        ax8.axhline(0, color='black', linewidth=0.5)
    except Exception as e:
        ax8.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', fontsize=6)

## 9. Correlation Matrix (Manual Implementation)
if len(numeric_cols) > 1:
    try:
        # Select most relevant numeric columns
        relevant_cols = []
        if 'Profit' in numeric_cols: relevant_cols.append('Profit')
        if 'COGS' in numeric_cols: relevant_cols.append('COGS')
        if 'Units Sold' in numeric_cols: relevant_cols.append('Units Sold')
        if 'Manufacturing Price' in numeric_cols: relevant_cols.append('Manufacturing Price')
        if 'Sale Price' in numeric_cols: relevant_cols.append('Sale Price')
        
        # Fill with other numeric columns if needed
        remaining_cols = [col for col in numeric_cols if col not in relevant_cols]
        relevant_cols.extend(remaining_cols[:5-len(relevant_cols)])
        
        if len(relevant_cols) > 1:
            corr = df[relevant_cols].corr().values
            
            # Create heatmap manually
            im = ax9.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax9, shrink=0.6)
            cbar.ax.tick_params(labelsize=6)
            
            # Add annotations
            for i in range(len(relevant_cols)):
                for j in range(len(relevant_cols)):
                    ax9.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=6)
            
            ax9.set_xticks(np.arange(len(relevant_cols)))
            ax9.set_yticks(np.arange(len(relevant_cols)))
            ax9.set_xticklabels(relevant_cols, rotation=90, fontsize=6)
            ax9.set_yticklabels(relevant_cols, fontsize=6)
            ax9.set_title('Key Metrics Correlation')
        else:
            ax9.text(0.5, 0.5, 'Not enough numeric columns', ha='center', va='center', fontsize=6)
    except:
        ax9.text(0.5, 0.5, 'Error in correlation', ha='center', va='center', fontsize=6)

plt.show()

```
###### 5) Add Title and save the dashboard .
