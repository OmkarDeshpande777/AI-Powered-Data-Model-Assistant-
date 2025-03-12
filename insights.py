import os
import secrets
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Ensure non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, session, send_file, jsonify, url_for, redirect
from io import BytesIO

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_dataset(df):
    """Perform basic dataset cleaning while removing unnamed columns."""
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]  # Remove unnamed columns
    df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical NaNs with mode
    df.fillna(df.mean(), inplace=True)  # Fill numerical NaNs with mean
    df.drop_duplicates(inplace=True)
    return df

def identify_relations(df):
    """Identify relations between numerical and categorical columns."""
    relations = []
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            relations.append(f"{num_cols[i]} vs {num_cols[j]}")
    
    for cat in cat_cols:
        for num in num_cols:
            relations.append(f"{cat} vs {num}")
    
    return relations

def generate_graph(df, relation):
    """Generate visually enhanced graphs based on relation type."""
    x, y = relation.split(" vs ")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    if x in df.select_dtypes(include=['object']).columns:
        sns.barplot(x=df[x], y=df[y], palette="coolwarm", ci=None)
        plt.xticks(rotation=30, fontsize=12)
    else:
        sns.scatterplot(x=df[x], y=df[y], s=80, alpha=0.6, edgecolors="black", color="royalblue")
        sns.regplot(x=df[x], y=df[y], scatter=False, color="crimson", line_kws={"linewidth": 2})
    
    plt.xlabel(x, fontsize=14, fontweight="bold")
    plt.ylabel(y, fontsize=14, fontweight="bold")
    plt.title(f"{x} vs {y}", fontsize=16, fontweight="bold", color="midnightblue")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def generate_insights(df, selected_relations):
    """Generate insights in plain English for each selected relation."""
    insights = {}
    for relation in selected_relations:
        x, y = relation.split(" vs ")
        
        if x in df.select_dtypes(include=['object']).columns:
            mean_value = df.groupby(x)[y].mean().to_dict()
            insights[relation] = f"The average {y} for each category in {x} is: {mean_value}."
        else:
            correlation = df[x].corr(df[y])
            trend = "increase" if correlation > 0 else "decrease"
            insights[relation] = f"The correlation between {x} and {y} is {correlation:.2f}. As {x} increases, {y} tends to {trend}."
    return insights

def generate_chart(filepath, x_col, y_col, chart_type, start_year=None, end_year=None):
    """Generate chart data for visualization."""
    df = pd.read_csv(filepath)
    if x_col not in df.columns or y_col not in df.columns:
        return {"error": f"Columns '{x_col}' or '{y_col}' not found in dataset"}
    
    df = df.dropna(subset=[x_col, y_col])
    df[x_col] = df[x_col].astype(str)
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    
    if start_year and end_year:
        try:
            start_year, end_year = int(start_year), int(end_year)
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df = df[(df[x_col] >= start_year) & (df[x_col] <= end_year)]
        except ValueError:
            pass
    
    df = df.dropna(subset=[y_col])
    labels = df[x_col].tolist()
    values = df[y_col].tolist()
    
    if not values:
        return {"labels": [], "values": [], "insights": "No data available."}
    
    max_value = df[y_col].max()
    min_value = df[y_col].min()
    avg_value = df[y_col].mean()
    max_index = df[y_col].idxmax()
    min_index = df[y_col].idxmin()
    
    insights = (
        f"The highest {y_col} was in {df.iloc[max_index][x_col]} with a value of {max_value}. "
        f"The lowest {y_col} was in {df.iloc[min_index][x_col]} with a value of {min_value}. "
        f"The average {y_col} is {round(avg_value, 2)}."
    )
    
    return {"labels": labels, "values": values, "insights": insights}

if __name__ == '__main__':
    app.run(debug=True)
