import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"C:\Users\harsd\OneDrive\Desktop\ML\ML_Assignment02_BL.EN.U4AIE23047\Lab Session Data.xlsx"  
xls = pd.ExcelFile(file_path)

# A1: Matrix Segregation and Analysis
def analyze_purchase_data():
    """
    Analyzes purchase data, calculates product costs, and returns relevant metrics.

    Returns:
        tuple: Dimensionality, number of vectors, rank of A, and product costs.
    """
    try:
        df = pd.read_excel(xls, sheet_name="Purchase data")
        purchase_matrix = df.iloc[:, 1:4].values  
        purchase_amounts = df.iloc[:, 4].values.reshape(-1, 1)  

        dimensionality = purchase_matrix.shape[1]
        num_vectors = purchase_matrix.shape[0]
        rank_A = np.linalg.matrix_rank(purchase_matrix)
        purchase_matrix_pinv = np.linalg.pinv(purchase_matrix)
        product_costs = np.dot(purchase_matrix_pinv, purchase_amounts).flatten()  

        print("A1 Results:")
        print(f"Dimensionality: {dimensionality}")
        print(f"Number of Vectors: {num_vectors}")
        print(f"Rank of A: {rank_A}")
        print(f"Product Costs: {product_costs}")
        return dimensionality, num_vectors, rank_A, product_costs
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None
    except ValueError:  # Catches potential Excel sheet issues
        print("Error: Could not read specified sheet from Excel file.")
        return None, None, None, None

# A2: Compute Model Vector X ( calls the A1 function)
def compute_model_vector():
    """Computes the model vector X (product costs)."""
    _, _, _, product_costs = analyze_purchase_data()  # Reuses A1's results
    if product_costs is not None:  # Check for potential errors from analyze_purchase_data
        print("A2 Result:")
        print(f"Model Vector X (Product Costs): {product_costs}")
        return product_costs
    else:
        return None

# A3: Customer Classification
def classify_customers():
    """Classifies customers as RICH or POOR based on purchase amount."""
    try:
        df = pd.read_excel(xls, sheet_name="Purchase data")
        df["Customer Class"] = np.where(df.iloc[:, 4] > 200, "RICH", "POOR")  
        print("A3 Result:")
        print(df[["Customer Class"]])
        return df[["Customer Class"]]
    except ValueError:
        print("Error: Could not read 'Purchase data' from Excel.")
        return None

# A4: IRCTC Stock Analysis
def analyze_irctc_stock():
    """Analyzes IRCTC stock data, calculates statistics, and generates visualizations."""
    try:
        df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day"] = df["Date"].dt.day_name()

        mean_price = statistics.mean(df["Price"])
        variance_price = statistics.variance(df["Price"])
        wednesday_mean = df[df["Day"] == "Wednesday"]["Price"].mean()  
        april_mean = df[df["Date"].dt.month == 4]["Price"].mean()  
        prob_loss = (df["Chg%"] < 0).mean()
        prob_profit_wed = df[(df["Day"] == "Wednesday") & (df["Chg%"] > 0)]["Chg%"].count() / df[df["Day"] == "Wednesday"]["Chg%"].count()

        print("A4 Results:")
        print(f"Mean Price: {mean_price}")
        print(f"Variance Price: {variance_price}")
        print(f"Wednesday Mean Price: {wednesday_mean}")
        print(f"April Mean Price: {april_mean}")
        print(f"Probability of Loss: {prob_loss}")
        print(f"Probability of Profit on Wednesday: {prob_profit_wed}")

        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=df["Day"], y=df["Chg%"])
        plt.xlabel("Day of the Week")  # axis labels
        plt.ylabel("Change %")
        plt.xticks(rotation=45)
        plt.title("Change % vs. Day of the Week")
        plt.tight_layout() #  prevents labels from overlapping
        plt.show()




        return mean_price, variance_price, wednesday_mean, april_mean, prob_loss, prob_profit_wed
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None, None, None
    except KeyError:  # Catch potential column name issues
        print("Error: One or more required columns ('Price', 'Chg%', 'Date') are missing from the Excel sheet.")
        return None, None, None, None, None, None
    except ValueError:
        print("Error: Could not read 'IRCTC Stock Price' from Excel.")
        return None, None, None, None, None, None

if __name__ == "__main__":
    analyze_purchase_data()
    compute_model_vector()
    classify_customers()
    analyze_irctc_stock()
