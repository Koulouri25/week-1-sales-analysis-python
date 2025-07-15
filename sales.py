import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your CSV file
csv_file_path = 'sales_data.csv'


try:
    # UPDATED LINE: Load the dataset, explicitly telling Pandas to use only certain columns
    # We will specify the column names that should exist based on your project description
    # Assuming your real columns are: Transaction_ID, Date, Customer_ID, Product, Category,
    # Price, Quantity, Total_Amount, Payment_Method, Region, Advertising_Spend
    df = pd.read_csv(csv_file_path, usecols=['Transaction_ID', 'Date', 'Customer_ID',
                                              'Product', 'Category', 'Quantity', 'Price',
                                              'Total_Amount', 'Payment_Method', 'Region',
                                              'Advertising_Spend'])

    print("Dataset loaded successfully!")
    print("\nFirst 5 rows of the dataset (after cleaning columns):")
    
    print("\nDataset Info (columns, non-null count, data types - after cleaning columns):")
    print(df.info())

    # Re-check for missing values (before handling)
    print("\nMissing values in each column (before handling):")
    print(df.isnull().sum())

    # --- Data Cleaning Steps ---

    # 1. Convert 'Date' column to datetime objects for proper time-series analysis
    # Use errors='coerce' to turn any problematic dates into NaT (Not a Time)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 2. Handle missing values
    # Fill missing numerical values with 0
    numerical_cols = ['Quantity', 'Price', 'Total_Amount', 'Advertising_Spend']
    for col in numerical_cols:
        df[col] = df[col].fillna(0)

    # Fill missing categorical/ID values with 'Unknown'
    categorical_id_cols = ['Customer_ID', 'Product', 'Category', 'Payment_Method', 'Region']
    for col in categorical_id_cols:
        df[col] = df[col].fillna('Unknown')
  
    #Handle Transaction_ID seperately
    df['Transaction_ID'] = df['Transaction_ID'].fillna(0).astype(int).astype(str)

    # For the 'Date' column, if any NaT resulted from coerce, fill with the mode (most frequent date)
    if df['Date'].isnull().any():
        mode_date = df['Date'].mode()[0]
        df['Date'] = df['Date'].fillna(mode_date)

    # Re-check for missing values (after handling them) to confirm
    print("\nMissing values in each column (after handling):")
    print(df.isnull().sum())

    # Check for duplicate rows
    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())

    #Remove duplicate rows
    df = df.drop_duplicates()

    # Get descriptive statistics for numerical columns
    print("\nDescriptive Statistics for Numerical Columns:")
    print(df.describe())

    # Check unique values in categorical columns
    print("\nUnique Products:")
    print(df['Product'].unique())

    print("\nUnique Categories:")
    print(df['Category'].unique())

    # Feature Engineering: Extract month and year for seasonal analysis
    df['Month'] = df['Date'].dt.month_name() # Get month name (e.g., 'March', 'April')
    df['Year'] = df['Date'].dt.year # Get year
    df['Month_Number'] = df['Date'].dt.month  # Get month number for correct sorting

    # Save the cleaned dataset to a new CSV file
    cleaned_csv_file_path = 'cleaned_sales_data.csv'
    df.to_csv(cleaned_csv_file_path, index=False) # index=False prevents writing the DataFrame index as a column

    print(f"\nCleaned dataset saved successfully to '{cleaned_csv_file_path}'")

    # --- Visualization 1: Total Sales by Product Category ---
    print("\nGenerating Total Sales by Category chart...")

    # Calculate total sales for each category
    category_sales = df.groupby('Category')['Total_Amount'].sum().sort_values(ascending=False)

    # Create the bar chart
    plt.figure(figsize=(10, 6)) # Sets the size of the chart
    sns.barplot(x=category_sales.index, y=category_sales.values) # Creates the bar plot
    plt.title('Total Sales by Product Category') # Adds a title to the chart
    plt.xlabel('Product Category') # Labels the X-axis
    plt.ylabel('Total Sales Amount') # Labels the Y-axis
    plt.xticks(rotation=45, ha='right') # Rotates category names for better readability
    plt.tight_layout() # Adjusts plot to ensure everything fits

    # Save the chart as an image file
    chart_path_category_sales = 'total_sales_by_category.png'
    plt.savefig(chart_path_category_sales)
    print(f"Chart saved to '{chart_path_category_sales}'")
    plt.close() # Closes the plot to free memory (important when generating many plots)

    # --- Visualization 2: Monthly Sales Trend ---
    print("\nGenerating Monthly Sales Trend chart...")

    # Group total sales by month
    monthly_sales = df.resample('ME', on='Date')['Total_Amount'].sum()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the chart
    plt.savefig('monthly_sales_trend.png')
    print("Chart saved to 'monthly_sales_trend.png'")
    plt.close()


    # --- Visualization 3: Correlation Heatmap ---
    numeric_cols = ['Quantity', 'Price', 'Total_Amount', 'Advertising_Spend']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
 
    
    # --- New Chart: Sales Distribution by Product (Pie Chart) ---
    print("\nGenerating Sales Distribution by Product chart...")

    # Group total sales by Product
    product_sales = df.groupby('Product')['Total_Amount'].sum()

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(product_sales.values, labels=product_sales.index, autopct='%1.1f%%', startangle=140)
    plt.title('Sales Distribution by Product')
    plt.axis('equal') # Ensures the pie is a circle

    # Save the chart
    plt.savefig('product_sales_distribution.png')
    print("Chart saved to 'product_sales_distribution.png'")
    plt.close()
 
    # --- Final Chart: Total Sales Over Time by Category ---
    print("\nGenerating Sales Trend by Category chart...")

    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Group by month and category, then sum the sales
    category_monthly_sales = df.groupby([df['Date'].dt.to_period('M'), 'Category'])['Total_Amount'].sum().unstack()

    # Plot
    category_monthly_sales.index = category_monthly_sales.index.to_timestamp()
    category_monthly_sales.plot(figsize=(10, 6), marker='o')
    plt.title('Sales Trend by Category Over Time')
    plt.xlabel('Month')
    plt.ylabel('Total Sales Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the chart
    plt.savefig('sales_trend_by_category.png')
    print("Chart saved to 'sales_trend_by_category.png'")
    plt.close()




except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please make sure 'sales_data.csv' is in the same folder as your script.")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")

else:
    print("\n All tasks completed successfully. Charts and cleaned data have been saved.")