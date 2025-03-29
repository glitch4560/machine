import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


file_path = r"C:\Users\harsd\OneDrive\Desktop\ML\ML_Assignment02_BL.EN.U4AIE23047\Lab Session Data.xlsx"  
xls = pd.ExcelFile(file_path)

# A5: Data Exploration
def explore_thyroid_data():
    """Explores the thyroid dataset, provides descriptive statistics, and identifies missing values."""
    try:
        df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
        df.replace('?', np.nan, inplace=True)
        df = df.infer_objects()  # Ensures proper type conversion
        missing_values = df.isnull().sum()

        # Converts categorical columns to string for Label Encoding 
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str)  # Converts to string
            df[col] = LabelEncoder().fit_transform(df[col])

        print("A5 Results:")
        print(df.describe())
        print("Missing Values:\n", missing_values)
        return df.describe(), missing_values

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except ValueError:  # Catches potential Excel sheet issues
        print("Error: Could not read specified sheet from Excel file.")
        return None, None

# A6: Data Imputation
def impute_missing_data():
    """Imputes missing values in the thyroid dataset using median for numerical and mode for categorical features."""
    try:
        df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
        df.replace('?', np.nan, inplace=True)
        df = df.infer_objects()

        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        print("A6 Results:")
        print(df)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError:  # Catches potential Excel sheet issues
        print("Error: Could not read specified sheet from Excel file.")
        return None


# A7: Data Normalization
def normalize_data():
    """Normalizes the thyroid dataset using MinMaxScaler after converting categorical features to numerical."""
    df = impute_missing_data()  # Uses the imputed data

    if df is None: # Handles potential errors from impute_missing_data
        return None

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print("A7 Results:")
    print(df)
    return df

# A8: Jaccard and SMC Similarity
def calculate_jaccard_smc():
    """Calculates the Jaccard Coefficient (JC) and Simple Matching Coefficient (SMC) between two rows."""
    df = normalize_data()
    if df is None:
        return None, None

    vector1 = df.iloc[0, :].values
    vector2 = df.iloc[1, :].values
    f11 = np.sum((vector1 == 1) & (vector2 == 1))
    f00 = np.sum((vector1 == 0) & (vector2 == 0))
    f10 = np.sum((vector1 == 1) & (vector2 == 0))
    f01 = np.sum((vector1 == 0) & (vector2 == 1))

    # Check for division by zero
    denominator = (f01 + f10 + f11)
    JC = f11 / denominator if denominator != 0 else 0  # Handles the case where all are 0
    SMC = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

    print("A8 Results:")
    print(f"Jaccard Coefficient: {JC}, SMC: {SMC}")
    return JC, SMC

# A9: Cosine Similarity
def calculate_cosine_similarity():
    """Calculates the cosine similarity between two rows."""
    df = normalize_data()
    if df is None:
        return None

    vector1 = df.iloc[0, :].values.reshape(1, -1)
    vector2 = df.iloc[1, :].values.reshape(1, -1)
    result = cosine_similarity(vector1, vector2)[0][0]
    print("A9 Result:", result)
    return result

# A10: Heatmap Plot for Similarity Measures (uses Euclidean distance as dissimilarity measure)
def plot_similarity_heatmap():
    """Generates a heatmap of Euclidean distances (dissimilarity) between the first 20 rows."""
    df = normalize_data()
    if df is None:
        return None

    df_subset = df.iloc[:20, :]  # Uses a subset for better visualization
    similarity_matrix = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            if i != j:
                similarity_matrix[i, j] = np.linalg.norm(df_subset.iloc[i] - df_subset.iloc[j])  # Euclidean distance

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm')  # (annot=False) for cleaner heatmap
    plt.title("Heatmap of Euclidean Distances (Dissimilarity)")
    plt.tight_layout()
    plt.show()
    print("A10 Result: (Euclidean Distance Matrix - Not Printed for brevity)")  # Not printing as it's a large matrix
    return similarity_matrix


if __name__ == "__main__":
    explore_thyroid_data()
    impute_missing_data()
    normalize_data()
    calculate_jaccard_smc()
    calculate_cosine_similarity()
    plot_similarity_heatmap()
