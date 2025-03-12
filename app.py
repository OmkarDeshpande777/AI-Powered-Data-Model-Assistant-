from flask import Flask, request, render_template, send_file, jsonify, session, url_for, redirect
import pandas as pd
from io import BytesIO
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv
import logging
from model_recommender import RobustNeuralModelSelector, generate_model_code
from insights import (
    clean_dataset,
    identify_relations,
    generate_graph,
    generate_insights
)
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for using sessions

# Global variables
cleaned_data = None
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("home_page.html")

@app.route("/data-cleaning", methods=["GET", "POST"])
def data_cleaning():
    global cleaned_data
    
    if request.method == "POST":
        if "dataset" in request.files:
            file = request.files["dataset"]
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
                column_list = df.columns.tolist()
                return render_template("data_cleaning.html", step="select_columns", columns=column_list, data=df.to_csv(index=False))

        elif "exclude" in request.form and "data" in request.form:
            exclude_columns = request.form["exclude"].split(",")
            df = pd.read_csv(BytesIO(request.form["data"].encode()))

            cleaned_data = clean_dataset(df, exclude_columns)
            return render_template("data_cleaning.html", step="preview", preview=cleaned_data.to_html(classes="table", border=1), data=cleaned_data.to_csv(index=False))

    return render_template("data_cleaning.html", step="upload")

@app.route("/download_cleaned", methods=["POST"])
def download_cleaned():
    global cleaned_data
    if cleaned_data is None:
        return "No cleaned dataset available to download.", 400
    
    output = BytesIO()
    cleaned_data.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="cleaned_dataset.csv")

def clean_dataset(df, exclude_columns):
    """
    Cleans the dataset by:
    - Removing selected columns (case-insensitive)
    - Dropping duplicate rows
    - Filling missing values (categorical: "Unknown", numerical: 0)
    """
    exclude_columns = [col.strip().lower() for col in exclude_columns]
    df.columns = df.columns.str.lower()

    df = df.drop(columns=exclude_columns, errors="ignore")
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)
    
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(0, inplace=True)
    
    return df

@app.route('/visualization_insights')
def visualization_insights():
    return render_template('viz_home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'dataset' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['dataset']
    if file.filename == '':
        return 'No file selected', 400

    if file and file.filename.endswith('.csv'):
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read and clean the dataset
        df = pd.read_csv(filepath)
        cleaned_df = clean_dataset(df, [])
        
        # Save cleaned dataset
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"cleaned_{file.filename}")
        cleaned_df.to_csv(cleaned_filepath, index=False)
        
        # Generate and store consistent top 5 relations
        relations = identify_relations(cleaned_df)
        top_relations = relations[:5] if relations else []  # Ensure no errors
        session['top_relations'] = top_relations  # Store in session
        
        # Generate graphs only for top 5 relations
        graph_images = {relation: generate_graph(cleaned_df, relation) for relation in top_relations}
        
        # Store filepath in session
        session['cleaned_filepath'] = cleaned_filepath
        
        return render_template('analysis.html', 
                             relations=graph_images, 
                             cleaned_filepath=cleaned_filepath)
    
    return 'Invalid file type', 400


@app.route('/generate_insights', methods=['POST'])
def generate_insights_route():
    filepath = request.form.get('cleaned_filepath')
    selected_relations = request.form.getlist('selected_relations')

    if not filepath or not selected_relations:
        return 'Missing required data', 400

    df = pd.read_csv(filepath)
    insights = generate_insights(df, selected_relations)

    # Ensure only the top 5 relations are used
    relations = identify_relations(df)[:5]  # Limit to top 5
    graph_images = {relation: generate_graph(df, relation) for relation in relations}

    return render_template('analysis.html',
                         relations=graph_images,
                         insights=insights,
                         cleaned_filepath=filepath)

@app.route("/customize_graph", methods=["GET", "POST"])
def customize_graph():
    # Get filepath from the query parameters
    filepath = request.args.get('cleaned_filepath')
    
    if not filepath:
        return redirect(url_for('home'))

    try:
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()
        year_columns = [col for col in df.columns if "year" in col.lower() or "date" in col.lower()]
        
        return render_template("customize.html", 
                             columns=columns, 
                             year_columns=year_columns,
                             cleaned_filepath=filepath)
    except Exception as e:
        logger.error(f"Error in customize_graph: {str(e)}")
        return redirect(url_for('home'))

@app.route("/generate_chart", methods=["POST"])
def generate_chart_route():
    try:
        data = request.get_json()
        logger.debug(f"Received chart request: {data}")
        
        # Validate required fields
        required_fields = ["cleaned_filepath", "x_col", "y_col", "chart_type"]
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        filepath = data["cleaned_filepath"]
        x_col = data["x_col"]
        y_col = data["y_col"]
        chart_type = data["chart_type"]
        start_year = data.get("start_year")
        end_year = data.get("end_year")

        if not os.path.exists(filepath):
            return jsonify({"error": f"File {filepath} not found"}), 400

        result = generate_chart(filepath, x_col, y_col, chart_type, start_year, end_year)
        
        # Check if result is a tuple (error case)
        if isinstance(result, tuple):
            return jsonify({"error": result[0]}), result[1]
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_chart(filepath, x_col, y_col, chart_type, start_year=None, end_year=None):
    try:
        df = pd.read_csv(filepath)

        # Filter by year range (if applicable)
        if start_year and end_year and "year" in df.columns:
            try:
                df = df[(df["year"] >= int(start_year)) & (df["year"] <= int(end_year))]
            except ValueError:
                return {"error": "Invalid year format"}, 400

        if x_col not in df.columns or y_col not in df.columns:
            return {"error": f"Columns {x_col} or {y_col} not found in dataset"}, 400

        # Handle empty dataframe
        if df.empty:
            return {"error": "No data available for the selected parameters"}, 400

        # Special handling for pie charts
        if chart_type == 'pie':
            # Group by the category column (x_col) and sum the values (y_col)
            try:
                grouped_data = df.groupby(x_col)[y_col].sum().reset_index()
                labels = grouped_data[x_col].astype(str).tolist()
                values = grouped_data[y_col].tolist()
            except Exception as e:
                return {"error": f"Error processing pie chart data: {str(e)}"}, 400
        else:
            # Regular charts
            labels = df[x_col].astype(str).tolist()
            values = df[y_col].tolist()

        # Generate basic insights
        try:
            avg_value = np.mean(values)
            insights = f"Average {y_col}: {avg_value:.2f}"
        except Exception as e:
            insights = "Unable to generate insights"

        return {
            "labels": labels,
            "values": values,
            "insights": insights
        }

    except pd.errors.EmptyDataError:
        return {"error": "The uploaded file is empty"}, 400
    except pd.errors.ParserError:
        return {"error": "Error parsing the CSV file"}, 400
    except Exception as e:
        logger.error(f"Error in generate_chart: {str(e)}")
        return {"error": f"Error processing data: {str(e)}"}, 500



@app.route('/model_recommendation', methods=['GET', 'POST'])
def model_recommendation():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                data = pd.read_csv(filepath)
                data.columns = data.columns.str.strip()

                if data.shape[1] < 2:
                    return jsonify({"error": "Dataset must have at least one feature and one target column."}), 400

                data.fillna(method='ffill', inplace=True)
                target_feature = data.columns[-1]
                X = data.drop(columns=[target_feature])
                y = data[target_feature]

                # Encode categorical features
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = LabelEncoder().fit_transform(X[col])

                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)

                X = X.values

                # Determine task type
                task_type = "classification" if len(np.unique(y)) <= 10 else "regression"
                
                selector = RobustNeuralModelSelector(task_type=task_type)
                selector.prepare_dataset(X, y)
                recommendations = selector.recommend_model_type(X)
                
                dataset_info = {
                    "features": list(data.columns[:-1]),
                    "target": target_feature,
                    "task_type": task_type
                }
                session['dataset_info'] = dataset_info
                session['recommendations'] = recommendations
                session['filepath'] = filepath

                return jsonify({
                    "recommendations": recommendations,
                    "features": dataset_info["features"],
                    "target": dataset_info["target"],
                    "task_type": dataset_info["task_type"]
                })

            except Exception as e:
                logger.exception("Error processing file")
                return jsonify({"error": str(e)}), 500

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        except Exception as e:
            logger.exception("Error in model recommendation")
            return jsonify({"error": str(e)}), 500

    return render_template('model.html')

@app.route('/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")

        if not data:
            return jsonify({"error": "No data provided"}), 400

        model_type = data.get("model")
        hyperparameters = data.get("hyperparameters", {})

        if not model_type:
            return jsonify({"error": "No model type specified"}), 400

        dataset_info = session.get('dataset_info')
        if not dataset_info:
            return jsonify({"error": "Dataset information not found. Please upload a file first."}), 400

        generated_code = generate_model_code(dataset_info, model_type, hyperparameters)
        logger.debug(f"Generated code for model {model_type} with hyperparameters {hyperparameters}")

        return jsonify({
            "status": "success",
            "generated_code": generated_code
        })

    except Exception as e:
        logger.exception("Error in generate_code endpoint")
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

if __name__ == "__main__":
    app.run(debug=True)