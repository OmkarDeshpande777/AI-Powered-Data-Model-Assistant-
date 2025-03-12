import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, session
from sklearn.preprocessing import StandardScaler, LabelEncoder
import google.generativeai as genai
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

# Load and configure API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No GOOGLE_API_KEY found in environment variables. Please set it.")
genai.configure(api_key=GOOGLE_API_KEY)

def get_dataset_metadata(data):
    """
    Generate metadata about the dataset including column types and basic statistics
    """
    metadata = {
        "n_rows": len(data),
        "n_columns": len(data.columns),
        "columns": {}
    }
    
    for column in data.columns:
        column_info = {
            "dtype": str(data[column].dtype),
            "n_unique": len(data[column].unique()),
            "n_missing": data[column].isnull().sum(),
            "is_numeric": pd.api.types.is_numeric_dtype(data[column])
        }
        
        if column_info["is_numeric"]:
            column_info.update({
                "mean": float(data[column].mean()) if not data[column].isnull().all() else None,
                "std": float(data[column].std()) if not data[column].isnull().all() else None,
                "min": float(data[column].min()) if not data[column].isnull().all() else None,
                "max": float(data[column].max()) if not data[column].isnull().all() else None
            })
        
        metadata["columns"][column] = column_info
    
    return metadata

def generate_model_code(dataset_info, model_type, hyperparameters, custom_prompt=None):
    """
    Generate code for the selected model using the Gemini API, incorporating dataset info and optional custom prompt
    """
    print(f"Generating complete code for the {model_type} model with hyperparameters...")

    hyperparam_str = "\n".join([f"    - {key}: {value}" for key, value in hyperparameters.items()])
    
    metadata_str = "\n".join([
        f"- Column '{col}': {info['dtype']}" 
        for col, info in dataset_info.get("metadata", {}).get("columns", {}).items()
    ])

    # Include sample data information
    sample_data_str = "\nSample Data Preview:\n"
    if dataset_info.get("sample_rows"):
        sample_data_str += str(pd.DataFrame(dataset_info["sample_rows"]).head().to_string())

    # Base prompt with dataset information
    base_prompt = f'''
You are an expert AI engineer. Your TASK is to generate Python code that performs the following tasks:
1. Load a dataset (CSV or Excel) into a Pandas DataFrame.
2. Preprocess the data, including handling missing values and encoding categorical features.
3. Split the data into training and testing sets.
4. Build and train a {model_type} model based on the preprocessed data.
5. The model should use the following hyperparameters:
{hyperparam_str}
6. Evaluate the model on the test set and print the results.
7. Include necessary imports for the model specified in the prompt.

Dataset Information:
- Features: {dataset_info["features"]}
- Target: {dataset_info["target"]}
- Task Type: {dataset_info["task_type"]}
Below is the metadata of the dataset, use it to make decision about the model code which columns to keep in features which columns
to keep in target and use the metadata to avoid any potential errors, drop whichever columns you feel are unecessary. Make sure the
generated code is tailored for the given dataset.
Dataset Metadata:
{metadata_str}

{sample_data_str}
'''

    # Combine base prompt with custom prompt if provided
    if custom_prompt:
        final_prompt = base_prompt + "\n\nAdditional User Requirements:\n" + custom_prompt
    else:
        final_prompt = base_prompt

    model = genai.GenerativeModel('gemini-1.0-pro')
    
    try:
        response = model.generate_content(final_prompt)
        generated_code = response.text
    except Exception as e:
        generated_code = f"Error: {str(e)}"

    return generated_code.strip()

# Define Default Hyperparameters
default_hyperparameters = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "criterion": "gini",
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 32,
    "hidden_layers": [64, 32],
    "activation": "relu",
    "n_estimators_xgb": 100,
    "max_depth_xgb": 6,
    "learning_rate_xgb": 0.1,
    "subsample": 1.0,
    "C_lr": 1.0,
    "solver": "lbfgs",
    "n_neighbors": 5,
    "weights": "uniform",
    "n_estimators_gb": 100,
    "learning_rate_gb": 0.1,
    "max_depth_gb": 3,
    "criterion_dt": "gini",
    "max_depth_dt": None,
}

class RobustNeuralModelSelector:
    def __init__(self, task_type="classification"):
        self.input_dim = None
        self.feature_extractor = None
        self.model_recommender = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.task_type = task_type
        self.num_classes = None

    def _validate_input(self, X, y=None):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.input_dim is None:
            self.input_dim = X.shape[1]

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input shape {X.shape} does not match expected {self.input_dim}")

        if y is not None:
            y = np.array(y)
            if len(X) != len(y):
                raise ValueError(f"Mismatch in input ({len(X)}) and label ({len(y)}) lengths")
            if self.task_type == "classification":
                y = self.label_encoder.fit_transform(y)

        return X, y

    def _build_feature_extractor(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def _build_model_recommender(self):
        inputs = tf.keras.Input(shape=(32,))
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        if self.task_type == "classification":
            if self.num_classes is None:
                raise ValueError("Number of classes not set for classification task.")
            output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)  # Dynamic output neurons
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            # For regression, output 3 values (one for each problem type)
            output = tf.keras.layers.Dense(3, activation='linear')(x)  # 3 outputs for regression
            loss = 'mean_squared_error'
            metrics = ['mean_squared_error']
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss=loss, metrics=metrics)
        return model

    def prepare_dataset(self, X, y=None):
        X, y = self._validate_input(X, y)
        X_scaled = self.scaler.fit_transform(X) if y is not None else self.scaler.transform(X)

        if self.feature_extractor is None:
            self.feature_extractor = self._build_feature_extractor()

        features = self.feature_extractor.predict(X_scaled)
        if y is not None:
            if self.task_type == "classification":
                self.num_classes = len(np.unique(y))  # Set number of classes
                y_categorical = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
                if self.model_recommender is None:
                    self.model_recommender = self._build_model_recommender()  # Build model with correct output neurons
                self.model_recommender.fit(features, y_categorical, epochs=50, validation_split=0.2, verbose=0)
            else:
                # For regression, reshape y to have 3 values per sample
                y_reshaped = np.tile(y.reshape(-1, 1), (1, 3))  # Repeat y 3 times to match model output
                if self.model_recommender is None:
                    self.model_recommender = self._build_model_recommender()
                self.model_recommender.fit(features, y_reshaped, epochs=50, validation_split=0.2, verbose=0)

        return features

    def recommend_model_type(self, X):
        X, _ = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        features = self.feature_extractor.predict(X_scaled)
        predictions = self.model_recommender.predict(features)

        if self.task_type == "classification":
            # Determine the number of problem types based on the model's output shape
            num_problem_types = predictions.shape[1]
            problem_types = ['Logistic Regression', 'Naïve Bayes', 'KNN'][:num_problem_types]
            return {problem_types[i]: float(predictions[0][i]) for i in range(num_problem_types)}
        else:
            # For regression, always use 3 problem types
            problem_types = ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression']
            return {problem_types[i]: float(predictions[0][i]) for i in range(len(problem_types))}

# Flask Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('model.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if not file or file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    try:
        file.save(filepath)

        try:
            data = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
            data.columns = data.columns.str.strip()

            if data.shape[1] < 2:
                return jsonify({"error": "Dataset must have at least one feature and one target column."}), 400

            # Get dataset metadata and sample rows
            metadata = get_dataset_metadata(data)
            sample_rows = data.head(5).to_dict(orient='records')

            data.fillna(method='ffill', inplace=True)
            target_feature = data.columns[-1]
            X = data.drop(columns=[target_feature])
            y = data[target_feature] if target_feature in data.columns else None

            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col])

            if y is not None and y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X = X.values

            # Determine task type
            if y is not None and len(np.unique(y)) <= 10:
                task_type = "classification"
            else:
                task_type = "regression"

            selector = RobustNeuralModelSelector(task_type=task_type)

            if y is not None:
                selector.prepare_dataset(X, y)

            recommendations = selector.recommend_model_type(X)
            top_models = sorted(recommendations, key=recommendations.get, reverse=True)[:3]

            dataset_info = {
                "features": list(data.columns[:-1]),
                "target": target_feature,
                "task_type": task_type,
                "metadata": metadata,
                "sample_rows": sample_rows
            }
            
            session['dataset_info'] = dataset_info
            session['recommendations'] = recommendations
            session['filepath'] = filepath

            return jsonify({
                "recommendations": recommendations,
                "features": dataset_info["features"],
                "target": dataset_info["target"],
                "task_type": dataset_info["task_type"],
                "metadata": metadata,
                "sample_rows": sample_rows
            })

        except pd.errors.EmptyDataError:
            return jsonify({"error": "The uploaded file is empty."}), 400
        except pd.errors.ParserError:
            return jsonify({"error": "Could not parse the file as CSV or Excel. Check the file format."}), 400
        except Exception as e:
            return jsonify({"error": f"An error occurred while reading the file: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        pass

@app.route('/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")

        if not data:
            return jsonify({"error": "No data provided"}), 400

        model_type = data.get("model")
        hyperparameters = data.get("hyperparameters", {})
        custom_prompt = data.get("custom_prompt", "")

        if not model_type:
            return jsonify({"error": "No model type specified"}), 400

        dataset_info = session.get('dataset_info')
        if not dataset_info:
            return jsonify({"error": "Dataset information not found. Please upload a file first."}), 400

        generated_code = generate_model_code(
            dataset_info, 
            model_type, 
            hyperparameters, 
            custom_prompt
        )
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

    finally:
        filepath = session.get('filepath')
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted uploaded file: {filepath}")
            session.pop('filepath', None)

if __name__ == '__main__':
    app.run(debug=True)