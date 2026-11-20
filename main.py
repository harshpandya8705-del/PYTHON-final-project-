"""
Pandas Analyzer & Data Visualization
A menu-driven sales analysis project using Pandas, NumPy, Matplotlib and Seaborn.
Run: python main.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

class SalesDataAnalyzer:
    """Encapsulates loading, cleaning, analysis and visualization of sales data."""

    def __init__(self, file_path=None):
        self.data = pd.DataFrame()
        if file_path:
            self.load_data(file_path)

    def __del__(self):
        # No external resources are held; destructor included to match the OOP brief.
        pass

    def load_data(self, file_path):
        path = Path(file_path)
        if not path.is_absolute():
            path = ROOT / path
        try:
            self.data = pd.read_csv(path)
            print(f"Dataset loaded successfully: {len(self.data)} rows.")
            return True
        except FileNotFoundError:
            print(f"File not found: {path}")
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            print(f"Could not read CSV: {exc}")
        except Exception as exc:
            print(f"Unexpected error while loading: {exc}")
        return False

    def explore_data(self):
        if not self._ready(): return
        print("\n--- First 5 rows ---")
        print(self.data.head())
        print("\n--- Last 5 rows ---")
        print(self.data.tail())
        print("\n--- Column names ---")
        print(list(self.data.columns))
        print("\n--- Data types ---")
        print(self.data.dtypes)
        print("\n--- Dataset information ---")
        self.data.info()
        print(f"\nShape: {self.data.shape}")

    def clean_data(self):
        if not self._ready(): return
        print("\nMissing values before cleaning:")
        print(self.data.isnull().sum())
        for col in self.data.select_dtypes(include=np.number).columns:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        for col in self.data.select_dtypes(include=["object"]).columns:
            if self.data[col].isnull().any():
                mode = self.data[col].mode()
                self.data[col] = self.data[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        if "Date" in self.data.columns:
            self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")
        if "Sales" in self.data.columns:
            self.data = self.data[self.data["Sales"] >= 0]
        self.data = self.data.drop_duplicates().reset_index(drop=True)
        print("\nCleaning completed. Missing values now:")
        print(self.data.isnull().sum())

    def mathematical_operations(self):
        if not self._ready(): return
        numeric = self.data.select_dtypes(include=np.number)
        if numeric.empty:
            print("No numeric columns available."); return
        print("\nNumeric column totals:\n", numeric.sum())
        print("\nNumeric column averages:\n", numeric.mean())
        if "Sales" in self.data:
            self.data["Sales_Tax_5pct"] = self.data["Sales"] * 0.05
            self.data["Sales_After_Tax"] = self.data["Sales"] + self.data["Sales_Tax_5pct"]
            print("\nAdded Sales_Tax_5pct and Sales_After_Tax columns.")

    def combine_data(self, other_dataframe):
        """Combine this dataset with another DataFrame using row-wise concatenation."""
        if self.data.empty: self.data = other_dataframe.copy()
        else: self.data = pd.concat([self.data, other_dataframe], ignore_index=True)
        return self.data

    def split_data(self, column="Region"):
        if not self._ready(): return {}
        if column not in self.data.columns:
            print(f"Column '{column}' does not exist."); return {}
        parts = {str(key): frame.copy() for key, frame in self.data.groupby(column, dropna=False)}
        for key, frame in parts.items():
            safe = "".join(ch if ch.isalnum() else "_" for ch in key)
            frame.to_csv(OUTPUT_DIR / f"split_{column}_{safe}.csv", index=False)
        print(f"Created {len(parts)} split file(s) in outputs/.")
        return parts

    def search_sort_filter(self):
        if not self._ready(): return
        print("\nAvailable columns:", ", ".join(self.data.columns))
        col = input("Column to search/filter: ").strip()
        if col not in self.data.columns:
            print("Unknown column."); return
        term = input("Value to search for (partial text is allowed): ").strip()
        mask = self.data[col].astype(str).str.contains(term, case=False, na=False)
        print("\nMatching records:\n", self.data.loc[mask].to_string(index=False))
        sort_col = input("Column to sort by (Enter to skip): ").strip()
        if sort_col:
            if sort_col not in self.data.columns:
                print("Unknown sort column."); return
            order = input("Ascending? (y/n): ").strip().lower() != "n"
            print("\nSorted data:\n", self.data.sort_values(sort_col, ascending=order).to_string(index=False))
        filter_col = input("Optional exact filter column (Enter to skip): ").strip()
        if filter_col:
            if filter_col not in self.data.columns:
                print("Unknown filter column."); return
            value = input("Exact value: ").strip()
            print("\nFiltered data:\n", self.data[self.data[filter_col].astype(str) == value].to_string(index=False))

    def aggregate_functions(self):
        if not self._ready(): return
        numeric = self.data.select_dtypes(include=np.number).columns
        print("\nNumeric aggregation (sum, mean, min, max, count):")
        print(self.data[numeric].agg(["sum", "mean", "min", "max", "count"]))
        if "Region" in self.data and "Sales" in self.data:
            print("\nSales by region:")
            print(self.data.groupby("Region")["Sales"].agg(["sum", "mean", "count"]))

    def statistical_analysis(self):
        if not self._ready(): return
        numeric = self.data.select_dtypes(include=np.number)
        print("\nDescriptive statistics:\n", numeric.describe())
        if "Sales" in self.data:
            print("\nSales standard deviation:", self.data["Sales"].std())
            print("Sales variance:", self.data["Sales"].var())
            print("Sales percentiles:\n", self.data["Sales"].quantile([.25, .5, .75, .9]))

    def create_pivot_table(self):
        if not self._ready(): return
        needed = {"Region", "Product", "Sales"}
        if not needed.issubset(self.data.columns):
            print("Pivot table requires Region, Product and Sales columns."); return
        pivot = pd.pivot_table(self.data, index="Region", columns="Product",
                               values="Sales", aggfunc="sum", fill_value=0, margins=True)
        print("\nSales pivot table:\n", pivot)
        pivot.to_csv(OUTPUT_DIR / "sales_pivot_table.csv")

    def visualize_data(self):
        if not self._ready(): return
        while True:
            print("""
--- Data Visualization ---
1. Bar Plot
2. Line Plot
3. Scatter Plot
4. Pie Chart
5. Histogram
6. Stack Plot
7. Seaborn Heatmap
8. Seaborn Box Plot
9. Generate all standard plots
0. Back to main menu""")
            choice = input("Enter your choice: ").strip()
            try:
                if choice == "0": return
                if choice == "1": self._bar_plot()
                elif choice == "2": self._line_plot()
                elif choice == "3": self._scatter_plot()
                elif choice == "4": self._pie_plot()
                elif choice == "5": self._histogram()
                elif choice == "6": self._stack_plot()
                elif choice == "7": self._heatmap()
                elif choice == "8": self._box_plot()
                elif choice == "9":
                    for method in (self._bar_plot, self._line_plot, self._scatter_plot,
                                   self._pie_plot, self._histogram, self._stack_plot,
                                   self._heatmap, self._box_plot):
                        try: method()
                        except (KeyError, ValueError) as exc: print("Skipped plot:", exc)
                else: print("Please choose a listed option.")
            except (KeyError, ValueError) as exc:
                print("Cannot create plot:", exc)

    def _save_plot(self, filename):
        path = OUTPUT_DIR / filename
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved: {path.relative_to(ROOT)}")

    def _bar_plot(self):
        if not {"Region", "Sales"}.issubset(self.data.columns): raise ValueError("Requires Region and Sales.")
        values = self.data.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        values.plot(kind="bar", title="Total Sales by Region", xlabel="Region", ylabel="Sales")
        self._save_plot("bar_sales_by_region.png")

    def _line_plot(self):
        if "Sales" not in self.data: raise ValueError("Requires Sales.")
        frame = self.data.copy()
        if "Date" in frame:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame = frame.dropna(subset=["Date"]).groupby("Date")["Sales"].sum().sort_index()
            frame.plot(kind="line", marker="o", title="Sales Trend", xlabel="Date", ylabel="Sales")
        elif "Year" in frame:
            frame.groupby("Year")["Sales"].sum().sort_index().plot(marker="o", title="Sales by Year", xlabel="Year", ylabel="Sales")
        else: raise ValueError("Requires Date or Year with Sales.")
        self._save_plot("line_sales_trend.png")

    def _scatter_plot(self):
        x = "Sales"
        y = "Profit" if "Profit" in self.data else ("Year" if "Year" in self.data else None)
        if not y: raise ValueError("Requires Profit or Year as the second numeric field.")
        plt.scatter(self.data[x], self.data[y], alpha=.75)
        plt.title(f"{x} vs {y}"); plt.xlabel(x); plt.ylabel(y)
        self._save_plot("scatter_sales_vs_" + y.lower() + ".png")

    def _pie_plot(self):
        if not {"Region", "Sales"}.issubset(self.data.columns): raise ValueError("Requires Region and Sales.")
        self.data.groupby("Region")["Sales"].sum().plot(kind="pie", autopct="%1.1f%%", ylabel="", title="Sales Share by Region")
        self._save_plot("pie_sales_by_region.png")

    def _histogram(self):
        if "Sales" not in self.data: raise ValueError("Requires Sales.")
        plt.hist(self.data["Sales"].dropna(), bins=10, edgecolor="black")
        plt.title("Distribution of Sales"); plt.xlabel("Sales"); plt.ylabel("Frequency")
        self._save_plot("histogram_sales.png")

    def _stack_plot(self):
        if not {"Region", "Product", "Sales"}.issubset(self.data.columns): raise ValueError("Requires Region, Product and Sales.")
        table = self.data.pivot_table(index="Product", columns="Region", values="Sales", aggfunc="sum", fill_value=0)
        table.plot(kind="bar", stacked=True, title="Stacked Sales by Product and Region", ylabel="Sales")
        self._save_plot("stacked_sales.png")

    def _heatmap(self):
        numeric = self.data.select_dtypes(include=np.number)
        if numeric.shape[1] < 2: raise ValueError("Heatmap requires at least two numeric columns.")
        plt.figure(figsize=(8, 5))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        self._save_plot("heatmap_correlation.png")

    def _box_plot(self):
        if not {"Region", "Sales"}.issubset(self.data.columns): raise ValueError("Requires Region and Sales.")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.data, x="Region", y="Sales")
        plt.title("Sales Distribution by Region")
        self._save_plot("boxplot_sales_by_region.png")

    def save_data(self):
        if not self._ready(): return
        path = OUTPUT_DIR / "cleaned_sales_data.csv"
        self.data.to_csv(path, index=False)
        print(f"Cleaned dataset saved to {path.relative_to(ROOT)}")

    def _ready(self):
        if self.data.empty:
            print("Please load a dataset first."); return False
        return True


def main():
    analyzer = SalesDataAnalyzer()
    while True:
        print("""
========== PANDAS ANALYZER & DATA VISUALIZATION ==========
1. Load Dataset
2. Explore Data
3. Perform DataFrame Operations
4. Handle Missing Data
5. Generate Descriptive Statistics
6. Data Visualization
7. Save Data / Pivot Table / Split by Region
8. Exit
==========================================================""")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            path = input("Enter CSV path (e.g. data/sales_data.csv): ").strip()
            analyzer.load_data(path)
        elif choice == "2": analyzer.explore_data()
        elif choice == "3":
            analyzer.mathematical_operations()
            analyzer.aggregate_functions()
            analyzer.create_pivot_table()
        elif choice == "4": analyzer.clean_data()
        elif choice == "5": analyzer.statistical_analysis()
        elif choice == "6": analyzer.visualize_data()
        elif choice == "7":
            analyzer.save_data()
            analyzer.create_pivot_table()
            analyzer.split_data("Region")
        elif choice == "8":
            print("Exiting the program. Goodbye!")
            break
        else: print("Invalid choice. Please enter 1-8.")

if __name__ == "__main__":
    main()
