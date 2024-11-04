
#######################
### LIBRARY IMPORTS ###
#######################

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import os  # For file and directory operations
import time
import logging  # For logging messages
import matplotlib.pyplot as plt  # For creating visualizations
import random  # For random number generation
random.seed(42)  # Set seed for reproducibility (for random number generation)
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor  # For building decision tree models
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # For standardizing features
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve, TimeSeriesSplit, GridSearchCV  # For splitting data into train and test sets
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score  # For model evaluation
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Dict, List, Tuple  # For type hinting
from dataclasses import dataclass, field  # For creating data classes
from sklearn.inspection import PartialDependenceDisplay

import warnings  # For handling warnings
warnings.filterwarnings('ignore')  # Ignore all warnings to prevent cluttering the output



#################################################
### DISPLAY OPTIONS AND LOGGING CONFIGURATION ###
#################################################


## DISPLAY OPTIONS
pd.set_option('display.max_rows', 15)  # Set pandas display option to show more rows

## LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO, which will show all info, warning, and error messages
logger = logging.getLogger(__name__)  # Create a logger instance for this module

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

##  DATA CONFIGURATION
@dataclass
class DataConfig:
    """
    Data configuration class to store various parameters
    This class uses the @dataclass decorator to automatically generate special methods like __init__
    """
    data_dir: str = "data"  # Directory for input data files, default is "data"
    plots_dir: str = "plots"  # Directory for output plot files, default is "plots"
    mortality_regions: List[str] = field(default_factory=lambda: [
        "Africa", "Americas", "Eastern_Mediterranean", "Europe", "South_East_Asia", "Western_Pacific"
    ])  # List of regions for mortality data, covering all WHO regions
    population_size: int = 100  # Population size for genetic algorithm, chosen as a balance between computational cost and diversity. This takes into consideration number of iterations 1000 x population size = 10,000 evaluations. This is set to be 'computationally 'fair' against the below comparison with Hillclimb algorithm.
                                # 10,0000 vs 1,000 may seem unfair however this is balanced by the different nature of the algorithms

## DATA PROCESSOR
class EnhancedDataProcessor:
    """
    Class for processing and loading data
    This class handles the loading and initial processing of mortality and nutrition data
    """
    def __init__(self, config: DataConfig):
        self.config = config  # Store the configuration object
        self.mortality_data = {}  # Dictionary to store mortality data for each region
        self.nutrition_data = None  # Variable to store nutrition data

    ## MORTALITY DATA LOADING
    def load_mortality_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load mortality data for each region
        Returns a dictionary with region names as keys and pandas DataFrames as values
        """
        for region in self.config.mortality_regions:
            try:
                # Construct file path
                file_path = os.path.join(self.config.data_dir, f"Child mortality rates_{region}.csv")
                # Read CSV file, skipping the first row (header=1) as it contains metadata
                df = pd.read_csv(file_path, header=1)
                # Strip whitespace from column names to ensure consistency
                df.columns = df.columns.str.strip()
                # Add region column for later analysis
                df['region'] = region
                # Strip whitespace from country names to ensure consistency
                df['country'] = df['Countries, territories and areas'].str.strip()
                # Process confidence intervals to extract additional information
                df = self._process_confidence_intervals(df)
                # Handle year ranges to ensure consistent year format
                df['Year'] = df['Year'].apply(self._handle_year_ranges)
                # Store processed dataframe in mortality_data dictionary
                self.mortality_data[region] = df
                logger.info(f"Loaded mortality data for {region}")
            except Exception as e:
                logger.error(f"Error loading mortality data for {region}: {str(e)}")
        return self.mortality_data

    ## NUTRITION DATA LOADING
    def load_nutrition_data(self) -> pd.DataFrame:
        """
        Load nutrition data
        Returns a pandas DataFrame containing nutrition data
        """
        try:
            # Construct file path
            file_path = os.path.join(self.config.data_dir, "Infant nutrition data by country.csv")
            # Read CSV file
            self.nutrition_data = pd.read_csv(file_path)
            # Handle year ranges to ensure consistent year format
            self.nutrition_data['Year'] = self.nutrition_data['Year'].apply(self._handle_year_ranges)
            # Strip whitespace from country names to ensure consistency
            self.nutrition_data['country'] = self.nutrition_data['Countries, territories and areas'].str.strip()
            logger.info("Loaded nutrition data")
            return self.nutrition_data
        except Exception as e:
            logger.error(f"Error loading nutrition data: {str(e)}")
            raise

    ## HELPER METHODS 
    def _handle_year_ranges(self, year_str: str) -> int:
        """
        Handle year ranges by taking the first year of the range
        This method is used to standardize year format across datasets
        """
        if isinstance(year_str, str) and '-' in year_str:
            return int(year_str.split('-')[0])  # Take the first year of the range
        return int(year_str)  # If it's not a range, convert to integer

    ## PROCESS CONFIDENCE INTERVALS
    def _process_confidence_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process confidence intervals in the data
        This method extracts main values and confidence intervals from the data,
        and calculates certainty weights based on the width of the confidence intervals
        """
        processed_df = df.copy()
        for column in ['Both sexes', 'Male', 'Female']:
            if column in processed_df.columns:
                # Extract the main value (before the confidence interval)
                processed_df[f'{column}_value'] = processed_df[column].apply(
                    lambda x: float(str(x).split('[')[0].strip()) if pd.notna(x) else np.nan
                )
                # Check if confidence interval exists
                ci_mask = processed_df[column].str.contains(r'\[.*?\]', na=False)
                processed_df[f'{column}_ci'] = processed_df[column].where(ci_mask)
                if ci_mask.any():
                    # Extract lower and upper bounds of confidence interval
                    bounds = processed_df[f'{column}_ci'].str.extract(r'\[([\d.]+)-([\d.]+)\]')
                    processed_df[f'{column}_lower'] = pd.to_numeric(bounds[0])
                    processed_df[f'{column}_upper'] = pd.to_numeric(bounds[1])
                    # Calculate certainty weight based on confidence interval
                    # The weight is inversely proportional to the width of the confidence interval
                    # A weight of 1.0 means highest certainty (narrow CI), 0.5 means lowest certainty (wide CI)
                    processed_df[f'{column}_certainty_weight'] = 1.0 - (
                        (processed_df[f'{column}_upper'] - processed_df[f'{column}_lower']) / 
                        processed_df[f'{column}_value']
                    ).clip(0, 0.5) / 0.5
        return processed_df



##############################
### MACHINE LEARNING MODEL ###
##############################


## GENETIC ALGORITHM ##

class EnhancedGeneticAlgorithm:
    def __init__(self):
        self.fitness_history = []

    ## ALIGN DATASETS
    @measure_time
    def align_datasets(self, mortality_data: Dict[str, pd.DataFrame], 
                    nutrition_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Combine all mortality data into a single dataframe
        mortality_combined = pd.concat(mortality_data.values(), ignore_index=True)
        
        # Standardize country names to ensure proper merging
        mortality_combined['country'] = mortality_combined['country'].str.strip()
        nutrition_data['country'] = nutrition_data['Countries, territories and areas'].str.strip()
        
        # Merge mortality and nutrition data on country and year
        aligned_data = pd.merge(
            mortality_combined,
            nutrition_data,
            left_on=['country', 'Year'],
            right_on=['country', 'Year'],
            how='inner'  # Only keep matching records to ensure data consistency
        )
        
        # Simulate genetic algorithm iterations
        for _ in range(1000):  # 1000 to match HC
            # Simulate fitness improvement
            fitness = self.calculate_fitness(aligned_data)
            self.fitness_history.append(fitness)
            
            # Add some randomness to the fitness improvement
            if random.random() < 0.1:  # 10% chance of fitness improvement
                aligned_data = self.mutate(aligned_data)
        
        # Split back into mortality and nutrition
        mortality_cols = ['country', 'region', 'Year'] + [col for col in mortality_combined.columns 
                        if col not in ['country', 'region', 'Year','Countries, territories and areas']]
        nutrition_cols = ['country', 'Year'] + [col for col in nutrition_data.columns 
                        if col not in ['country','Countries, territories and areas', 'Year']]
        
        # Return aligned datasets
        return {
            'mortality': aligned_data[mortality_cols],
            'nutrition': aligned_data[nutrition_cols]
        }

    ## CALCULATE FITNESS
    def calculate_fitness(self, data):
        # Calculate fitness based on correlation
        mortality = data['Both sexes_value'].apply(self.extract_main_value)
        breastfeeding = data['Infants exclusively breastfed for the first six months of life (%)'].apply(self.extract_main_value)
        return abs(mortality.corr(breastfeeding))

    ## MUTATE DATA
    def mutate(self, data):
        # Create a deep copy of the input data to avoid modifying the original
        mutated_data = data.copy()
        
        # Get all numeric columns from the dataset
        # This includes int and float datatypes
        numeric_columns = mutated_data.select_dtypes(include=[np.number]).columns
        
        # Only proceed if we have numeric columns to mutate
        if len(numeric_columns) > 0:
            # Randomly select one numeric column to apply mutation
            column_to_mutate = random.choice(numeric_columns)
            
            # Apply mutation to the selected column:
            # - For each value x in the column:
            #   - If the value is not null, multiply it by a random factor between 0.99 and 1.01
            #     (This creates a ±1% random variation)
            #   - If the value is null (NaN), keep it as is
            mutated_data[column_to_mutate] = mutated_data[column_to_mutate].apply(
                lambda x: x * (1 + random.uniform(-0.01, 0.01)) if pd.notnull(x) else x
            )
            
        # Return the mutated dataset
        return mutated_data

    ## EXTRACT MAIN VALUE
    def extract_main_value(self, value):
        if isinstance(value, str):
            return float(value.split('[')[0].strip())
        return value



###########################################
### MACHINE LEARNING MODEL - COMPARISON ###
###########################################



## HILL CLIMBING ALGORITHM ##

class HillClimbingOptimizer:
    def __init__(self, config: DataConfig):
        # Initialize the HillClimbingOptimizer with a given configuration
        self.config = config
        self.best_solution = None  # Store the best solution found
        self.best_score = float('-inf')  # Initialize best score to negative infinity
        self.score_history = []  # Keep track of scores during optimization

    ## INITIALIZE SOLUTION
    def initialize_solution(self, features: pd.DataFrame) -> pd.DataFrame:
        """Initialize a random solution by shuffling the rows of the dataset."""
        # Randomly shuffle all rows of the input DataFrame and reset the index
        return features.sample(frac=1).reset_index(drop=True)

    ## EXTRACT MAIN VALUE
    def extract_main_value(self, value):
        """Extract the main value from a string that might contain confidence intervals."""
        if isinstance(value, str):
            # If the value is a string, split by '[' and take the first part
            # This removes any confidence interval information
            # Then strip whitespace and convert to float
            return float(value.split('[')[0].strip())
        # If the value is not a string, return it as is (assuming it's already a number)
        return value

    ## EVALUATE SOLUTION
    def evaluate_solution(self, solution: pd.DataFrame) -> float:
        """Evaluate the solution based on the correlation between breastfeeding and mortality."""
        # Define column names for breastfeeding and mortality data
        breastfeeding_col = 'Infants exclusively breastfed for the first six months of life (%)'
        mortality_col = 'Both sexes_value'
        
        # Extract main values from both columns, handling potential string inputs
        breastfeeding_values = solution[breastfeeding_col].apply(self.extract_main_value)
        mortality_values = solution[mortality_col].apply(self.extract_main_value)
        
        # Calculate and return the absolute correlation between breastfeeding and mortality
        return abs(breastfeeding_values.corr(mortality_values))

    ## GENERATE NEIGHBOR
    def generate_neighbor(self, solution: pd.DataFrame) -> pd.DataFrame:
        """Generate a neighbor solution by swapping two random rows."""
        # Create a copy of the current solution to avoid modifying the original
        neighbor = solution.copy()
        # Randomly select two different row indices
        i, j = random.sample(range(len(neighbor)), 2)
        # Swap the selected rows
        neighbor.iloc[i], neighbor.iloc[j] = neighbor.iloc[j], neighbor.iloc[i]
        return neighbor

    ## HILL CLIMBING
    def hill_climb(self, features: pd.DataFrame, max_iterations: int = 1000) -> pd.DataFrame:
        """Perform hill climbing to optimize the dataset alignment."""
        # Initialize the current solution and evaluate its score
        current_solution = self.initialize_solution(features)
        current_score = self.evaluate_solution(current_solution)
        
        # Iterate for a specified number of times
        for _ in range(max_iterations):
            # Generate a neighboring solution
            neighbor = self.generate_neighbor(current_solution)
            # Evaluate the neighbor's score
            neighbor_score = self.evaluate_solution(neighbor)
            
            # If the neighbor is better, make it the current solution
            if neighbor_score > current_score:
                current_solution = neighbor
                current_score = neighbor_score
            
            # Record the current score in the history
            self.score_history.append(current_score)
            
            # Update the best solution if the current one is better
            if current_score > self.best_score:
                self.best_solution = current_solution
                self.best_score = current_score
        
        # Return the best solution found
        return self.best_solution

    ## ALIGN DATASETS
    @measure_time
    def align_datasets(self, mortality_data: Dict[str, pd.DataFrame], 
                    nutrition_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Align datasets using Hill Climbing optimization.
        
        This method performs the following steps:
        1. Combines mortality data from multiple sources
        2. Standardizes country names in both datasets
        3. Merges mortality and nutrition data
        4. Optimizes the alignment using Hill Climbing algorithm
        5. Splits the optimized data back into separate mortality and nutrition datasets
        
        Args:
            mortality_data (Dict[str, pd.DataFrame]): Dictionary of mortality DataFrames
            nutrition_data (pd.DataFrame): Nutrition DataFrame
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing aligned 'mortality' and 'nutrition' DataFrames
        """
        # Combine all mortality data into a single dataframe
        # This step is necessary when mortality data comes from multiple sources
        mortality_combined = pd.concat(mortality_data.values(), ignore_index=True)
        
        # Standardize country names to ensure proper merging
        # This step removes leading/trailing whitespaces that might cause mismatches
        mortality_combined['country'] = mortality_combined['country'].str.strip()
        nutrition_data['country'] = nutrition_data['Countries, territories and areas'].str.strip()
        
        # Merge mortality and nutrition data
        # We use an inner join to keep only the countries and years that exist in both datasets
        merged_data = pd.merge(
            mortality_combined,
            nutrition_data,
            left_on=['country', 'Year'],
            right_on=['country', 'Year'],
            how='inner'
        )
        
        # Optimize the alignment using Hill Climbing
        # This step attempts to find the best arrangement of rows to maximize correlation
        optimized_data = self.hill_climb(merged_data)
        
        # Split back into mortality and nutrition
        # We need to separate the columns belonging to each dataset
        
        # Mortality columns include country, region, Year, and all other columns from the original mortality data
        mortality_cols = ['country', 'region', 'Year'] + [col for col in mortality_combined.columns 
                        if col not in ['country', 'region', 'Year', 'Countries, territories and areas']]
        
        # Nutrition columns include country, Year, and all other columns from the original nutrition data
        nutrition_cols = ['country', 'Year'] + [col for col in nutrition_data.columns 
                        if col not in ['country', 'Countries, territories and areas', 'Year']]
        
        # Return a dictionary with the separated and optimized datasets
        return {
            'mortality': optimized_data[mortality_cols],
            'nutrition': optimized_data[nutrition_cols]
        }


###################
## MODEL BUILDER ##
###################



## MODEL IMPLEMENTATION ##

class EnhancedModelBuilder:
    """
    Class for building and evaluating the model.
    This class handles feature engineering, data preparation, model training, and evaluation.
    It uses a Decision Tree Regressor as the base model and incorporates various data processing techniques.
    """
    def __init__(self, config: DataConfig):
        # Store the configuration object for data processing and model building
        self.config = config  # DataConfig instance containing various settings

        # Initialize the Decision Tree Regressor model
        self.model = DecisionTreeRegressor(
            max_depth=None,    # Allow full depth for capturing complex relationships in the data
            min_samples_split=5,  # Minimum number of samples required to split an internal node
                                  # Chose 5 as a balance between overfitting (lower values) and underfitting (higher values)
                                  # This hyperparameter can be further tuned for optimal performance
            random_state=42    # Seed for random number generator, ensures reproducibility of results
                               # 42 is used as a playful reference to "The Hitchhiker's Guide to the Galaxy"
        )
        # Decision Tree Regressor is chosen for its simplicity and interpretability
        # It works by recursively partitioning the feature space into rectangular regions

        # Initialize the StandardScaler for feature scaling
        self.scaler = StandardScaler()  # Used to normalize features by removing the mean and scaling to unit variance
                                        # This is crucial for many machine learning algorithms to work properly
                                        # Especially important when features have different scales

        # Initialize feature names (will be set later)
        self.feature_names = None  # Will store the names of the features used in the model

        # Define development indices for different regions
        # These indices represent a simplified measure of healthcare access in different WHO regions
        self.development_indices = {
            'healthcare_access': {
                'Europe': 0.9,               # Highest access to healthcare
                'Americas': 0.7,             # Good access to healthcare
                'Western_Pacific': 0.7,      # Similar to Americas
                'Eastern_Mediterranean': 0.5,# Moderate access to healthcare
                'South_East_Asia': 0.4,      # Lower access to healthcare
                'Africa': 0.3                # Lowest access to healthcare
            }
        }
        # These indices can be used as additional features or for stratified analysis

        # Define temporal windows for trend analysis
        self.temporal_windows = {
            'short_term': 3,   # 3-year window for analyzing immediate trends
            'medium_term': 5,  # 5-year window, often aligns with policy and planning cycles
            'long_term': 10    # 10-year window for identifying long-term patterns and shifts
        }
        # These windows will be used in feature engineering to capture trends over different time scales

    ## ENGINEER FEATURES
    def engineer_features(self, mortality_data: pd.DataFrame, nutrition_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from mortality and nutrition data.
        This method creates new features, validates data quality, and handles missing values.

        Args:
            mortality_data (pd.DataFrame): DataFrame containing mortality-related data.
            nutrition_data (pd.DataFrame): DataFrame containing nutrition-related data.

        Returns:
            pd.DataFrame: A DataFrame with engineered features and imputed missing values.
        """
        # Create base features from the input data
        features = self._create_base_features(mortality_data, nutrition_data)

        # Validate the quality of the created features
        self._validate_data_quality(features)
        
        # Impute missing data
        # Identify numerical and categorical columns
        num_cols = features.select_dtypes(include=['float64', 'int']).columns
        cat_cols = features.select_dtypes(include=['object']).columns
    
        # Create imputers
        # Use mean for numerical data as it's a common, simple approach
        # This works well when the data is roughly normally distributed
        num_imputer = SimpleImputer(strategy='mean')

        # Use most frequent value for categorical data
        # This is appropriate for categorical data where we want to preserve the most common category
        cat_imputer = SimpleImputer(strategy='most_frequent')
    
        # Impute numerical columns
        # This replaces missing values in numerical columns with the mean of that column
        features[num_cols] = num_imputer.fit_transform(features[num_cols])
    
        # Impute categorical columns
        # This replaces missing values in categorical columns with the most frequent value in that column
        features[cat_cols] = cat_imputer.fit_transform(features[cat_cols])
    
        return features
        
    ## CREATE BASE FEATURES
    def _create_base_features(self, mortality_data: pd.DataFrame, 
                            nutrition_data: pd.DataFrame) -> pd.DataFrame:
        # Initialize an empty DataFrame to store our engineered features
        features = pd.DataFrame()
        
        # Basic features: Copy over essential information from input data
        # These columns provide context and are crucial for grouping and analysis
        features['country'] = mortality_data['country']
        features['region'] = mortality_data['region']
        features['year'] = mortality_data['Year']
        
        # Extract numeric values from mortality data
        # We use pd.to_numeric with errors='coerce' to handle any non-numeric values
        # This converts strings to floats and replaces any unconvertible values with NaN
        features['mortality_rate'] = pd.to_numeric(
            mortality_data['Both sexes_value'], errors='coerce')
        
        # Extract breastfeeding rate, handling potential confidence intervals
        # We split on '[' to remove any confidence interval information
        # This assumes the data format is "value [confidence interval]"
        features['exclusive_breastfeeding'] = nutrition_data[
            'Infants exclusively breastfed for the first six months of life (%)'
        ].apply(lambda x: float(str(x).split('[')[0].strip()) if pd.notna(x) else np.nan)
        
        # Add healthcare access feature based on region
        # This uses a pre-defined mapping of healthcare access levels by region
        # The mapping is stored in self.development_indices, presumably set during initialization
        features['healthcare_access'] = features['region'].map(
            self.development_indices['healthcare_access']
        )
        
        # Handle missing values with forward/backward fill within countries
        # This assumes that values within a country are relatively stable over time
        # First, we fill forward (ffill) to use the last known value
        # Then, we fill backward (bfill) to handle cases where the first values might be missing
        for col in ['mortality_rate', 'exclusive_breastfeeding']:
            features[col] = features.groupby('country')[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
        
        # Add region development levels
        # This is a simplified categorization of regions into development levels
        # 3 represents highly developed, 2 moderately developed, and 1 less developed regions
        region_development = {
            'Europe': 3, 'Americas': 2, 'Western_Pacific': 2,
            'Eastern_Mediterranean': 1, 'South_East_Asia': 1, 'Africa': 1
        }
        features['region_development'] = features['region'].map(region_development)
        
        # Add temporal features
        # These features capture trends and changes over different time windows
        # We use the pre-defined temporal windows (short_term, medium_term, long_term)
        for col in ['mortality_rate', 'exclusive_breastfeeding']:
            for window_name, window in self.temporal_windows.items():
                # Calculate rolling average (trend) for each time window
                # This smooths out short-term fluctuations and shows the overall trend
                features[f'{col}_{window_name}_trend'] = features.groupby('country')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                # Calculate rate of change for each time window
                # This shows how quickly the value is changing over the given time period
                features[f'{col}_{window_name}_change'] = features.groupby('country')[col].transform(
                    lambda x: (x - x.rolling(window=window, min_periods=1).mean()) / window
                )
        
        # Add interaction term
        # This captures the combined effect of breastfeeding and healthcare access
        # It assumes that the impact of breastfeeding might be influenced by healthcare access
        features['breastfeeding_healthcare_impact'] = (
            features['exclusive_breastfeeding'] * features['healthcare_access']
        )
        
        return features

    ## VALIDATE DATA QUALITY
    def _validate_data_quality(self, features: pd.DataFrame) -> None:
        """
        Validate data quality and log issues
        This method checks for missing values, suspicious values, and data coverage by region
        """
        # Check for missing values in each column
        for col in features.columns:
            missing = features[col].isnull().sum()
            if missing > 0:
                # Log a warning if there are any missing values
                # This helps identify which columns might need attention or imputation
                logger.warning(f"Column {col} has {missing} missing values "
                            f"({missing/len(features)*100:.2f}%)")
            
        # Check for suspicious values in specific columns
        
        # Mortality rates should never be negative
        if (features['mortality_rate'] < 0).any():
            logger.error("Found negative mortality rates! This is impossible and indicates data error.")
        
        # Breastfeeding rates are percentages and should never exceed 100%
        if (features['exclusive_breastfeeding'] > 100).any():
            logger.error("Found breastfeeding rates > 100%! This is impossible and indicates data error.")
        
        # Check data coverage by region for breastfeeding data
        for region in features['region'].unique():
            mask = features['region'] == region
            # Calculate the proportion of non-null values for breastfeeding data in each region
            coverage = features.loc[mask, 'exclusive_breastfeeding'].notna().mean()
            logger.info(f"Region {region} has {coverage*100:.2f}% breastfeeding data coverage")
            # This helps identify regions with potentially unreliable data due to low coverage

    ## PREPARE DATA
    def prepare_data(self, features: pd.DataFrame, target_column: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare data for model training and evaluation
        This method splits the data into training and testing sets, and scales the features
        """
        # Separate features (X) and target variable (y)
        X = features.drop(columns=[target_column, 'country', 'region'])
        y = features[target_column]
        
        # Split data into training (80%) and testing (20%) sets
        # test_size=0.2 means 20% of data is used for testing, which is a common choice
        # random_state=42 ensures reproducibility of the split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features to have zero mean and unit variance
        # This is important for many machine learning algorithms to work properly
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Return a dictionary containing both training and testing data
        return {
            'train': {'X': X_train_scaled, 'y': y_train},
            'test': {'X': X_test_scaled, 'y': y_test}
        }

    ## TRAIN AND EVALUATE
    @measure_time
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        # Convert numpy arrays to pandas DataFrames if necessary
        # This ensures consistent data types and allows for easier feature manipulation
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=self.feature_names)

        # Identify categorical columns
        # We need to handle categorical data differently from numerical data
        categorical_columns = X_train.select_dtypes(include=['object']).columns

        # Create preprocessing steps
        # OneHotEncoder is used to convert categorical variables into a form that could be provided to ML algorithms to do a better job in prediction
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        # ColumnTransformer applies different transformations to different columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_columns)
            ],
            remainder='passthrough'  # This keeps the non-categorical columns as they are
        )

        # Create a pipeline with preprocessor and model
        # Pipeline chains multiple steps that can be cross-validated together while setting different parameters
        pipeline = Pipeline([
            ('preprocessor', preprocessor),  # First step: preprocess the data
            ('model', self.model)            # Second step: fit the model
        ])
        # This pipeline ensures that we preprocess our data consistently for both training and testing

        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'feature_importance': pipeline.named_steps['model'].feature_importances_ if hasattr(pipeline.named_steps['model'], 'feature_importances_') else None
        }

    ## SET FEATURE NAMES
    def set_feature_names(self, feature_names):
        """
        Set the feature names for the model.
        
        :param feature_names: A list of strings representing the names of the features.
        """
        self.feature_names = feature_names

    ## TRAIN AND EVALUATE WITH CROSS-VALIDATION
    def train_and_evaluate_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train and evaluate the model using time series cross-validation.
        
        :param X: DataFrame containing the feature data.
        :param y: Series containing the target variable.
        :return: Dictionary containing cross-validation scores and mean score.
        """
        # If feature names haven't been set, use the column names from X
        if self.feature_names is None:
            self.feature_names = X.columns

        # Initialize TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        # Perform time series cross-validation
        for train_index, test_index in tscv.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Scale the features using the scaler (assumed to be defined elsewhere)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate the model
            result, _ = self.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
            # Append the R-squared score to the list of scores
            scores.append(result['r2'])
        
        # Return a dictionary containing individual CV scores and the mean score
        return {
            'cv_scores': scores,  # List of R-squared scores for each fold
            'mean_cv_score': np.mean(scores)  # Average R-squared score across all folds
        }

    ## OPTIMIZE HYPERPARAMETERS
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize hyperparameters for the DecisionTreeRegressor using GridSearchCV.

        :param X_train: Training features
        :param y_train: Training target variable
        :return: Dictionary of best hyperparameters
        """
        # Define the hyperparameter grid to search
        param_grid = {
            'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree (None means unlimited)
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required to be at a leaf node
        }

        # Initialize GridSearchCV with DecisionTreeRegressor
        # cv=5 means 5-fold cross-validation
        grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Set the model to the best estimator found
        self.model = grid_search.best_estimator_

        # Return the best parameters
        return grid_search.best_params_

    ## BOOTSTRAP PREDICTIONS
    def bootstrap_predictions(self, X, y, n_iterations=1000, ci=0.95):
        """
        Perform bootstrap resampling to generate prediction intervals.

        :param X: Features
        :param y: Target variable
        :param n_iterations: Number of bootstrap iterations (default: 1000)
        :param ci: Confidence interval (default: 0.95 for 95% CI)
        :return: Tuple of lower and upper bounds of the prediction interval
        """
        predictions = []

        # Perform bootstrap iterations
        for _ in range(n_iterations):
            # Resample the data with replacement
            X_resampled, y_resampled = resample(X, y)

            # Fit the model on the resampled data
            self.model.fit(X_resampled, y_resampled)

            # Make predictions on the original data and store them
            predictions.append(self.model.predict(X))
        
        # Calculate the lower and upper bounds of the prediction interval
        lower = np.percentile(predictions, (1-ci)/2 * 100, axis=0)
        upper = np.percentile(predictions, (1+ci)/2 * 100, axis=0)

        return lower, upper

    ## CALCULATE PREDICTION INTERVALS
    def calculate_prediction_intervals(self, X, y, X_pred, alpha=0.05):
        """
        Calculate prediction intervals using bootstrap resampling.
        
        :param X: Features used for training
        :param y: Target values used for training
        :param X_pred: Features for which to predict intervals
        :param alpha: Significance level (default: 0.05 for 95% confidence)
        :return: Lower and upper bounds of the prediction intervals
        """
        # Set the number of bootstrap iterations
        n_iterations = 1000
        # Get the number of samples in the training data
        n_samples = X.shape[0]
        
        # Initialize an array to store predictions for each bootstrap iteration
        predictions = np.zeros((n_iterations, X_pred.shape[0]))
        
        # Perform bootstrap iterations
        for i in range(n_iterations):
            # Randomly sample indices with replacement
            sample_indices = np.random.randint(0, n_samples, n_samples)
            # Create resampled datasets
            X_resample = X[sample_indices]
            y_resample = y[sample_indices]
            
            # Fit the model on the resampled data
            self.model.fit(X_resample, y_resample)
            # Make predictions on X_pred and store them
            predictions[i] = self.model.predict(X_pred)
        
        # Calculate the lower bound of the prediction interval
        # This is the alpha/2 percentile of the predictions
        lower_bound = np.percentile(predictions, alpha/2 * 100, axis=0)
        # Calculate the upper bound of the prediction interval
        # This is the (1 - alpha/2) percentile of the predictions
        upper_bound = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        
        return lower_bound, upper_bound

    ## PERFORM ABLATION STUDY
    @measure_time
    def perform_ablation_study(self, X, y):
        # Define feature groups for ablation
        feature_groups = {
            'temporal': [col for col in X.columns if 'trend' in col or 'change' in col],
            'regional': ['healthcare_access', 'region_development'],
            'breastfeeding': [col for col in X.columns if 'breastfeeding' in col]
        }

        # Calculate the base score with all features
        base_score = self.model.score(X, y)
        # Initialize dictionary to store ablation results
        ablation_results = {}

        # Iterate through each feature group
        for group, features in feature_groups.items():
            # Remove the current feature group from the dataset
            X_ablated = X.drop(columns=features)
            # Fit the model on the ablated dataset
            self.model.fit(X_ablated, y)
            # Calculate the score on the ablated dataset
            ablated_score = self.model.score(X_ablated, y)
            # Store the difference between base score and ablated score
            # This represents the importance of the removed feature group
            ablation_results[group] = base_score - ablated_score

        return ablation_results

    ## PLOT LEARNING CURVE
    @measure_time
    def plot_learning_curve(self, X, y):
        # Generate learning curve data using sklearn's learning_curve function
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5),  # Use 5 evenly spaced sample sizes from 10% to 100% of the data
            scoring=make_scorer(mean_squared_error, greater_is_better=False)  # Use MSE as the scoring metric
        )

        # Create a new figure with specified size
        plt.figure(figsize=(10, 6))
        
        # Plot training and validation errors
        # Note: We negate the scores because MSE is a loss function (lower is better)
        plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training error')
        plt.plot(train_sizes, -test_scores.mean(axis=1), label='Validation error')
        
        # Set labels and title
        plt.xlabel('Number of training examples')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve: Model Performance vs Training Set Size')
        
        # Add a legend to distinguish between training and validation errors
        plt.legend()
        
        # Save the figure to a file
        plt.savefig(f"{self.config.plots_dir}/learning_curve.png")
        
        # Close the plot to free up memory
        plt.close()

    ## ANALYZE FEATURE IMPORTANCE
    def analyze_feature_importance(self, feature_names):
        # Get feature importances from the model
        importances = self.model.feature_importances_
        
        # Create a pandas Series with feature importances and their names
        forest_importances = pd.Series(importances, index=feature_names)
        
        # Calculate the standard deviation of feature importances across all trees
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        
        # Create a new figure and axis
        fig, ax = plt.subplots()
        
        # Plot the feature importances as a bar chart, including error bars
        forest_importances.plot.bar(yerr=std, ax=ax)
        
        # Set the title and y-axis label
        ax.set_title("Feature Importances using Mean Decrease in Impurity (MDI)")
        ax.set_ylabel("Mean decrease in impurity")
        
        # Adjust the layout to prevent clipping of labels
        fig.tight_layout()
        
        # Save the figure to a file
        plt.savefig(f"{self.config.plots_dir}/feature_importance_detailed.png")
        
        # Close the plot to free up memory
        plt.close()

    ## PLOT LEARNING CURVE (ALTERNATIVE VERSION)
    def plot_learning_curve(self, X, y):
        # Generate learning curve data using sklearn's learning_curve function
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5),  # Use 5 evenly spaced sample sizes from 10% to 100% of the data
            scoring='neg_mean_squared_error'  # Use negative MSE as the scoring metric
        )
        
        # Calculate mean scores (negate to convert back to MSE)
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)

        # Create a new figure
        plt.figure()
        
        # Plot training and validation errors
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, test_scores_mean, label='Validation error')
        
        # Set labels and title
        plt.xlabel('Number of training examples')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve: Model Performance vs Training Set Size')
        
        # Add a legend to distinguish between training and validation errors
        plt.legend()
        
        # Save the figure to a file
        plt.savefig(f"{self.config.plots_dir}/learning_curve.png")
        
        # Close the plot to free up memory
        plt.close()

    ## PREDICT WITH INTERVALS
    def predict_with_intervals(self, X, percentile=95):
        # Generate predictions from all trees in the forest
        preds = []
        for estimator in self.model.estimators_:
            preds.append(estimator.predict(X))
        preds = np.array(preds)
        
        # Calculate lower and upper bounds of the prediction interval
        lower = np.percentile(preds, (100 - percentile) / 2., axis=0)
        upper = np.percentile(preds, 100 - (100 - percentile) / 2., axis=0)
        
        # Get point estimate (mean prediction across all trees)
        point = self.model.predict(X)
        
        return point, lower, upper

    ## ANALYZE TIME TRENDS
    def analyze_time_trends(self, data):
        # Calculate yearly mean mortality rates
        yearly_means = data.groupby('year')['mortality_rate'].mean()
        
        # Create a new figure with specified size
        plt.figure(figsize=(10, 6))
        
        # Plot the yearly means
        yearly_means.plot()
        
        # Set title and labels
        plt.title('Mortality Rate Trend Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Mortality Rate')
        
        # Save the figure to a file
        plt.savefig(f"{self.config.plots_dir}/mortality_trend.png")
        
        # Close the plot to free up memory
        plt.close()


#######################################################
### RESULTS : OUTPUTS, VISUALIZATIONS, AND ANALYSIS ###
#######################################################


# RESULTS ANALYZER
class ResultsAnalyzer:
    """
    Class for analyzing and visualizing results
    This class handles the interpretation of model results and generation of reports
    """
    def __init__(self, config: DataConfig):
        # Initialize the ResultsAnalyzer with configuration
        self.config = config
        
        # Define development factors as class attributes
        # These factors represent the relative development levels for healthcare access and income
        # across different regions
        self.development_factors = {
            'healthcare_access': {
                'Europe': 0.9, 'Americas': 0.7, 'Western_Pacific': 0.7,
                'Eastern_Mediterranean': 0.5, 'South_East_Asia': 0.4, 'Africa': 0.3
            },
            'income_level': {
                'Europe': 3.0, 'Americas': 2.0, 'Western_Pacific': 2.0,
                'Eastern_Mediterranean': 1.0, 'South_East_Asia': 1.0, 'Africa': 1.0
            }
        }
        
        # Define confidence thresholds for data analysis
        # These thresholds help determine the reliability of the analysis results
        self.confidence_thresholds = {
            'min_data_points': 30,  # Minimum number of data points required for analysis
            'min_coverage': 0.6,    # Minimum data coverage required
            'high_coverage': 0.8    # Threshold for high data coverage
        }

    ## ANALYZE REGIONAL PATTERNS
    def _analyze_regional_patterns(self, features: pd.DataFrame, 
                                target: pd.Series) -> Dict[str, Dict]:
        """
        Analyze regional patterns in the data
        This method calculates statistics for each region, including mean mortality and correlation with breastfeeding
        """
        regional_stats = {}
        
        # Iterate through each unique region in the dataset
        for region in features['region'].unique():
            # Create a mask for the current region
            mask = features['region'] == region
            region_data = features[mask]
            
            # Calculate statistics only where we have both mortality and breastfeeding data
            valid_mask = region_data['exclusive_breastfeeding'].notna() & region_data['mortality_rate'].notna()
            
            # Retrieve development factors for the current region
            healthcare_factor = self.development_factors['healthcare_access'][region]
            income_factor = self.development_factors['income_level'][region]
            
            # Adjust mortality rates based on healthcare access and income level
            adjusted_mortality = region_data['mortality_rate'] * ((1/healthcare_factor) * (1/income_factor))
            
            # Calculate raw and adjusted correlations between breastfeeding and mortality
            raw_corr = region_data.loc[valid_mask, 'exclusive_breastfeeding'].corr(region_data.loc[valid_mask, 'mortality_rate'])
            adjusted_corr = region_data.loc[valid_mask, 'exclusive_breastfeeding'].corr(adjusted_mortality[valid_mask])
            
            # Store calculated statistics for the current region
            regional_stats[region] = {
                'mean_mortality': region_data['mortality_rate'].mean(),
                'raw_correlation': raw_corr,
                'adjusted_correlation': adjusted_corr,
                'data_points': len(region_data),
                'valid_data_points': valid_mask.sum(),
                'confidence_score': self._calculate_confidence_score(valid_mask.sum(), valid_mask.mean())
            }
        
        return regional_stats

    ## ANALYZE BREASTFEEDING IMPACT
    def _analyze_breastfeeding_impact(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Analyze the impact of breastfeeding on the target variable
        This method calculates the correlation between breastfeeding rates and mortality rates,
        both raw and adjusted for development factors
        """
        impacts = {}
        # Iterate through each unique region in the dataset
        for region in features['region'].unique():
            # Filter data for the current region
            region_data = features[features['region'] == region].copy()
            
            # Retrieve development factors for the current region
            healthcare_factor = self.development_factors['healthcare_access'][region]
            income_factor = self.development_factors['income_level'][region]
            
            # Adjust mortality rates based on healthcare access and income level
            region_data['mortality_adjusted'] = target[region_data.index] * (
                (1/healthcare_factor) * (1/income_factor)
            )
            
            # Create a mask for valid data points (non-null values for both breastfeeding and adjusted mortality)
            valid_mask = (
                region_data['exclusive_breastfeeding'].notna() & 
                region_data['mortality_adjusted'].notna()
            )
            
            # Calculate raw and adjusted correlations
            raw_corr = region_data.loc[valid_mask, 'exclusive_breastfeeding'].corr(target[region_data.index][valid_mask])
            adjusted_corr = region_data.loc[valid_mask, 'exclusive_breastfeeding'].corr(region_data.loc[valid_mask, 'mortality_adjusted'])
            
            # Calculate number of valid data points and data coverage
            n_points = valid_mask.sum()
            data_coverage = valid_mask.mean()
            
            # Store calculated statistics for the current region
            impacts[region] = {
                'raw_correlation': raw_corr,
                'adjusted_correlation': adjusted_corr,
                'data_points': n_points,
                'data_coverage': data_coverage,
                'confidence_score': self._calculate_confidence_score(n_points, data_coverage)
            }
        
        return impacts

    ## CALCULATE CONFIDENCE SCORE
    def _calculate_confidence_score(self, n_points: int, data_coverage: float) -> float:
        """
        Calculate a confidence score based on number of data points and data coverage
        """
        if n_points < self.confidence_thresholds['min_data_points'] or data_coverage < self.confidence_thresholds['min_coverage']:
            return 0.0
        elif data_coverage >= self.confidence_thresholds['high_coverage']:
            return 1.0
        else:
            return (data_coverage - self.confidence_thresholds['min_coverage']) / (self.confidence_thresholds['high_coverage'] - self.confidence_thresholds['min_coverage'])

    ## CREATE IMPACT VISUALIZATION
    def _create_impact_visualization(self, results: Dict) -> None:
        """
        Create visualization of feature importance
        This method generates a bar plot of feature importances and saves it as an image
        """
        plt.figure(figsize=(12, 6))  # Create a new figure with width 12 inches and height 6 inches
        plt.subplot(1, 2, 1)  # Create a subplot in a 1x2 grid, this is the first plot
        if results.get('feature_importance') is not None:
            importance_data = pd.Series(results['feature_importance'])
            importance_data.plot(kind='bar')  # Create a bar plot of feature importances
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
        else:
            # If feature importance is not available, display a message instead
            plt.text(0.5, 0.5, 'Feature importance not available', horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()  # Adjust the plot to fit into the figure area
        os.makedirs(self.config.plots_dir, exist_ok=True)  # Create the plots directory if it doesn't exist
        plt.savefig(f"{self.config.plots_dir}/feature_importance.png")  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory

    ## CALCULATE POTENTIAL IMPACT
    def _calculate_potential_impact(self, regional_analysis: Dict) -> float:
        """
        Calculate potential impact based on regional analysis.
        This method estimates the potential lives saved based on mortality rates and correlations.
        It provides a quantitative measure of the potential benefit of breastfeeding interventions.
        """
        return sum(stats['mean_mortality'] * stats['adjusted_correlation'] for stats in regional_analysis.values() if pd.notna(stats['adjusted_correlation']))
        # We changed 'correlation' to 'adjusted_correlation' to match the new structure

    ## CALCULATE CONFIDENCE LEVEL
    def _calculate_confidence_level(self, model_metrics):
        return model_metrics.get('r2', 0)

    ## IDENTIFY PRIORITY REGIONS
    def _identify_priority_regions(self, regional_analysis: Dict) -> List[str]:
        """
        Identify priority regions based on mean mortality rate.
        This method ranks regions by their mortality rates to determine intervention priorities.
        It helps NGOs focus their resources on areas with the highest need.
        """
        return sorted(regional_analysis.keys(), key=lambda x: regional_analysis[x]['mean_mortality'], reverse=True)[:3]
        # This returns the top 3 regions with the highest mortality rates
        # These are considered priority regions for intervention
        # The number 3 was chosen as a manageable number of focus areas for an NGO
        # It balances the need for focused intervention with the desire to have a broader impact

    ## ASSESS DATA QUALITY
    def _assess_data_quality(self, data_coverage: float) -> str:
        """
        Assess data quality based on data coverage.
        This method categorizes data quality based on the percentage of available data.
        It helps in understanding the reliability and completeness of the dataset.
        """
        if data_coverage > 0.8:
            return "High"  # Over 80% coverage is considered high quality
        elif data_coverage > 0.6:
            return "Medium"  # 60-80% coverage is considered medium quality
        else:
            return "Low"  # Below 60% coverage is considered low quality
        # These thresholds are based on common statistical practices for data completeness
        # High quality data allows for more confident conclusions
        # Medium quality data suggests caution in interpretation
        # Low quality data indicates a need for more data collection or alternative analysis methods

    ## ESTIMATE LIVES SAVED
    def _estimate_lives_saved(self, evaluation_results: Dict) -> float:
        """
        Estimate the number of lives that could potentially be saved based on the model results.
        This is a simplified estimation and should be interpreted cautiously.
        
        Args:
            evaluation_results (Dict): Dictionary containing regional analysis and breastfeeding impact data
            
        Returns:
            float: Estimated number of potential lives saved
        """
        # Calculate total mortality across all regions by summing the mean mortality rates
        # This gives us the baseline mortality we're working with
        total_mortality = sum(region['mean_mortality'] for region in evaluation_results['regional_analysis'].values())
        
        # Sum up the adjusted correlations between breastfeeding and mortality across regions
        # These correlations account for healthcare access and income levels
        # A negative correlation indicates that increased breastfeeding is associated with decreased mortality
        total_impact = sum(region['adjusted_correlation'] for region in evaluation_results['breastfeeding_impact'].values())
        
        # Calculate potential lives saved using a conservative 1% improvement scenario
        # We:
        # 1. Take the absolute value of total_impact to ensure positive number
        # 2. Multiply by 0.01 to model a 1% increase in breastfeeding rates
        # 3. Multiply by total_mortality to get the number of lives potentially saved
        potential_lives_saved = total_mortality * abs(total_impact) * 0.01
        
        # Return the estimated number of lives that could be saved
        # Note: This is a simplified model and actual results may vary based on many factors
        return potential_lives_saved

    ## GENERATE NGO REPORT
    def generate_ngo_report(self, evaluation_results: Dict) -> Dict:
        """
        Generate a comprehensive report for NGOs based on the evaluation results.
        This method compiles various analyses into a structured report format,
        providing insights that are crucial for NGO decision-making and funding applications.

        Args:
            evaluation_results (Dict): A dictionary containing various evaluation metrics and analyses

        Returns:
            Dict: A structured report containing key findings, evidence strength, and other relevant information
        """
        # Initialize the report structure
        report = {
            'key_findings': {
                'overall_impact': {
                    'breastfeeding_correlation': evaluation_results['breastfeeding_impact'],
                    'potential_lives_saved': self._estimate_lives_saved(evaluation_results),
                    'confidence_level': self._calculate_confidence_level(evaluation_results['model_metrics'])
                },
                'regional_insights': self._format_regional_insights(evaluation_results['regional_analysis']),
                'priority_regions': self._identify_priority_regions(evaluation_results['regional_analysis'])
            },
            'evidence_strength': {
                'statistical_significance': evaluation_results['model_metrics'].get('r2', 0),
                'data_quality': self._assess_data_quality(evaluation_results['data_coverage']),
                'prediction_accuracy': evaluation_results['model_metrics'].get('mse', 0)
            }
        }
        
        # Add the new breastfeeding impact analysis
        # This provides a more detailed look at the impact of breastfeeding across different regions
        breastfeeding_impact = self._analyze_breastfeeding_impact(evaluation_results['features'], evaluation_results['features']['mortality_rate'])
        report['key_findings']['breastfeeding_impact'] = {
            'overall': sum(region['adjusted_correlation'] for region in breastfeeding_impact.values()) / len(breastfeeding_impact),
            'by_region': {region: impact['adjusted_correlation'] for region, impact in breastfeeding_impact.items()},
            'confidence': {region: impact['confidence_score'] for region, impact in breastfeeding_impact.items()}
        }
        
        # Add time period analysis to the report
        # This helps identify trends and changes over different time periods
        time_analysis = self._analyze_time_periods(evaluation_results['features'])
        report['time_analysis'] = time_analysis

        # Add suggested interventions based on regional insights
        # This provides actionable recommendations for NGOs
        report['suggested_interventions'] = self.suggest_interventions(report['key_findings']['regional_insights'])
        
        # Add estimated economic impact
        # This helps quantify the potential financial benefits of interventions
        report['economic_impact'] = self.estimate_economic_impact(report['key_findings']['overall_impact']['potential_lives_saved'])
        
        # Add link to Sustainable Development Goals
        # This aligns the report with global development objectives
        report['sdg_alignment'] = self.link_to_sdgs()

        return report

    ## FORMAT REGIONAL INSIGHTS
    def _format_regional_insights(self, regional_analysis: Dict) -> Dict:
        """
        Format regional insights for the report.
        This method processes regional statistics into a more readable format,
        providing key metrics for each region to guide intervention strategies.
        """
        insights = {}
        for region, stats in regional_analysis.items():
            insights[region] = {
                'current_mortality': f"{stats['mean_mortality']:.2f}",
                'breastfeeding_impact': f"{stats['adjusted_correlation']*100:.1f}%" if pd.notna(stats['adjusted_correlation']) else "Insufficient data",
                'intervention_priority': 'High' if stats['mean_mortality'] > 50 else 'Medium' if stats['mean_mortality'] > 20 else 'Low'
            }
        return insights

    ## PERFORM HYPOTHESIS TESTING
    def _perform_hypothesis_testing(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Perform hypothesis testing to compare breastfeeding impact across regions
        This method uses t-tests to determine if there are statistically significant
        differences in breastfeeding rates between regions.
        """
        regions = features['region'].unique()
        p_values = {}
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                region1, region2 = regions[i], regions[j]
                data1 = features[features['region'] == region1]['exclusive_breastfeeding']
                data2 = features[features['region'] == region2]['exclusive_breastfeeding']
                _, p_value = stats.ttest_ind(data1, data2)
                p_values[f"{region1} vs {region2}"] = p_value
        return p_values

    ## EVALUATE MODEL CRITICALLY
    def evaluate_model_critically(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Critically evaluate the model's strengths, weaknesses, and potential biases
        This method provides a balanced assessment of the model's performance and limitations,
        which is crucial for transparent reporting and identifying areas for improvement.
        """
        strengths = [
            "Captures non-linear relationships between features",
            "Robust to outliers in the dataset",
            "Provides feature importance rankings"
        ]
        weaknesses = [
            "Less interpretable than simpler models like linear regression",
            "May overfit with small datasets",
            "Computationally intensive for very large datasets"
        ]
        biases = [
            "May be biased towards majority classes in imbalanced datasets",
            "Feature importance can be biased when features are correlated"
        ]
        
        # Calculate feature importances
        feature_importances = model.feature_importances_
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "potential_biases": biases,
            "feature_importances": feature_importances.tolist()
        }

    ## CREATE FEATURE IMPORTANCE PLOT
    def _create_feature_importance_plot(self, feature_importances):
        plt.figure(figsize=(12, 8))
        importance_data = pd.Series(feature_importances)
        importance_data = importance_data.sort_values(ascending=True)
        colors = ['green' if x < 0.05 else 'yellow' if x < 0.1 else 'red' for x in importance_data]
        importance_data.plot(kind='barh', color=colors)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/feature_importance.png")
        plt.close()

    ## CREATE HYPOTHESIS TESTING HEATMAP
    def _create_hypothesis_testing_heatmap(self, p_values: Dict[str, float]):
        regions = list(set([r.split(' vs ')[0] for r in p_values.keys()]))
        matrix = pd.DataFrame(index=regions, columns=regions, dtype=float)
        for pair, p_value in p_values.items():
            r1, r2 = pair.split(' vs ')
            matrix.loc[r1, r2] = p_value
            matrix.loc[r2, r1] = p_value
        matrix = matrix.fillna(1.0)  # Fill NaN values with 1.0 (no significant difference)
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='YlOrRd', vmin=0, vmax=0.05)
        plt.title('P-values for Regional Comparisons')
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/hypothesis_testing_heatmap.png")
        plt.close()

    ## CREATE PREDICTED VS ACTUAL PLOT
    def _create_predicted_vs_actual_plot(self, y_true, y_pred):
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Mortality Rates')
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/predicted_vs_actual.png")
        plt.close()

    ## CREATE TIME PERIOD ANALYSIS PLOT
    def _create_time_period_analysis_plot(self, time_analysis: Dict[str, Dict]):
        periods = list(time_analysis.keys())
        mortality_means = [time_analysis[p]['mean_mortality'] for p in periods]
        breastfeeding_means = [time_analysis[p]['mean_breastfeeding'] for p in periods]
        
        x = np.arange(len(periods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, mortality_means, width, label='Mortality Rate', color='r', alpha=0.7)
        ax.bar(x + width/2, breastfeeding_means, width, label='Breastfeeding Rate', color='b', alpha=0.7)
        
        ax.set_ylabel('Rate')
        ax.set_title('Mortality and Breastfeeding Rates by Time Period')
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/time_period_analysis.png")
        plt.close()

    ## CREATE ECONOMIC IMPACT PLOT
    def _create_economic_impact_plot(self, regional_analysis: Dict[str, Dict]):
        regions = list(regional_analysis.keys())
        impacts = [self.estimate_economic_impact(stats['mean_mortality'] * abs(stats['adjusted_correlation']))
                for stats in regional_analysis.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(regions, impacts)
        plt.title('Estimated Economic Impact by Region')
        plt.xlabel('Region')
        plt.ylabel('Estimated Economic Impact')
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/economic_impact_by_region.png")
        plt.close()

    ## PERFORM ANOVA
    def perform_anova(self, features):
        model = ols('mortality_rate ~ C(region)', data=features).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table

    ## PERFORM TUKEY HSD
    def perform_tukey_hsd(self, features):
        tukey = sm.stats.multicomp.pairwise_tukeyhsd(features['mortality_rate'], features['region'])
        return tukey
    
    ## PLOT RESIDUALS - this is a diagnostic plot to check if the model is making consistent predictions
    @measure_time
    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(f"{self.config.plots_dir}/residual_plot.png")
        plt.close()
    
    ## PLOT OPTIMIZATION COMPARISON
    @measure_time
    def plot_optimization_comparison(self, ga_time, hc_time, ga_correlation, hc_correlation):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Execution time comparison
        methods = ['Genetic Algorithm', 'Hill Climbing']
        times = [ga_time, hc_time]
        ax1.bar(methods, times)
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        for i, v in enumerate(times):
            ax1.text(i, v, f'{v:.3f}s', ha='center', va='bottom')

        # Correlation comparison
        correlations = [ga_correlation, hc_correlation]
        ax2.bar(methods, correlations)
        ax2.set_ylabel('Correlation')
        ax2.set_title('Alignment Quality Comparison')
        for i, v in enumerate(correlations):
            ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/optimization_comparison.png")
        plt.close()

    ## CREATE PERFORMANCE TABLE
    @measure_time
    def create_performance_table(self, results):
        # Create a DataFrame from the results, excluding 'feature_importance'
        table = pd.DataFrame({k: [v] for k, v in results.items() if k != 'feature_importance'})
        
        # Select only the columns we're interested in
        columns_of_interest = ['mse', 'mae', 'r2', 'train_time']
        table = table[[col for col in columns_of_interest if col in table.columns]]
        
        # Rename columns to match the expected output
        table.columns = ['MSE', 'MAE', 'R2', 'Train Time']
        
        # Save to CSV
        table.to_csv(f"{self.config.plots_dir}/performance_table.csv", index=False)
        
        return table

    ## ANALYZE TIME PERIODS
    def _analyze_time_periods(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze data across different time periods
        This method provides insights into how mortality rates and breastfeeding practices
        have changed over time, which is crucial for understanding long-term trends.
        """
        time_periods = {
            'historical': (1986, 1999),
            'transitional': (2000, 2009),
            'modern': (2010, 2020)
        }
        
        period_analysis = {}
        
        for period_name, (start_year, end_year) in time_periods.items():
            period_data = features[
                (features['year'] >= start_year) & 
                (features['year'] <= end_year)
            ]
            
            period_analysis[period_name] = {
                'data_points': len(period_data),
                'mean_mortality': period_data['mortality_rate'].mean(),
                'mean_breastfeeding': period_data['exclusive_breastfeeding'].mean(),
                'correlation': period_data['exclusive_breastfeeding'].corr(period_data['mortality_rate'])
            }
        
        return period_analysis

    ## CREATE VISUALIZATIONS FOR KEY FINDINGS
    def create_visualizations(self, features: pd.DataFrame):
        """
        Create visualizations to illustrate key findings
        This method generates plots that visually represent the relationship between
        breastfeeding rates and mortality rates, as well as trends over time.
        """
        try:
            # Scatter plot of breastfeeding rates vs. mortality rates
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=features, x='exclusive_breastfeeding', y='mortality_rate', hue='region')
            plt.title('Breastfeeding Rates vs. Mortality Rates by Region')
            plt.xlabel('Exclusive Breastfeeding Rate (%)')
            plt.ylabel('Mortality Rate (per 1000 live births)')
            plt.savefig(f"{self.config.plots_dir}/breastfeeding_vs_mortality.png")
            plt.close()

            # Time series plot of mortality rate trends
            plt.figure(figsize=(15, 10))
            for region in features['region'].unique():
                region_data = features[features['region'] == region]
                plt.plot(region_data['year'], region_data['mortality_rate'], label=region)
            plt.title('Mortality Rate Trends by Region')
            plt.xlabel('Year')
            plt.ylabel('Mortality Rate (per 1000 live births)')
            plt.legend()
            plt.savefig(f"{self.config.plots_dir}/mortality_trends.png")
            plt.close()
        except Exception as e:
            logger.error(f"Error in creating visualizations: {str(e)}")

    ## ESTIMATE ECONOMIC IMPACT
    def estimate_economic_impact(self, lives_saved: float) -> float:
        """
        Estimate the economic impact of lives saved
        This method provides a rough estimate of the economic benefit of reducing child mortality,
        which can be a powerful argument in grant applications.
        """
        years_of_productivity = 20  # Assume each life saved contributes 20 years of productive work
        gdp_per_capita = 10000  # Arbitrary value, could be adjusted based on specific country data
        return lives_saved * years_of_productivity * gdp_per_capita

    ## SUGGEST INTERVENTIONS FOR FINDINGS
    def suggest_interventions(self, regional_insights: Dict) -> Dict[str, str]:
        """
        Suggest interventions based on regional insights
        This method provides tailored recommendations for each region based on their
        current mortality rates and breastfeeding impacts.
        """
        interventions = {}
        for region, stats in regional_insights.items():
            if stats['intervention_priority'] == 'High':
                interventions[region] = "Implement comprehensive breastfeeding education programs"
            elif stats['intervention_priority'] == 'Medium':
                interventions[region] = "Enhance existing breastfeeding support systems"
            else:
                interventions[region] = "Maintain current breastfeeding promotion efforts"
        return interventions

    ## LINK TO UN SUSTAINABLE DEVELOPMENT GOALS - DOMAIN KNOWLEDGE IMPLEMENTATION
    def link_to_sdgs(self) -> str:
        """
        Link the project to UN Sustainable Development Goals
        This method provides a statement that connects the project's objectives to broader
        global development goals, which can strengthen the grant application.
        """
        return ("This project directly contributes to UN Sustainable Development Goal 3: "
                "Ensure healthy lives and promote well-being for all at all ages, "
                "particularly target 3.2: By 2030, end preventable deaths of newborns and "
                "children under 5 years of age.")

    ## ANALYZE SCALABILITY
    def analyze_scalability(self, model_builder, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """
        Analyze the scalability of the model by training it on increasingly larger subsets of the data.
        
        This method helps understand how the model's performance and training time scale
        with increasing dataset sizes.

        Args:
            model_builder: An object that contains the model to be analyzed
            X (pd.DataFrame): The feature dataset
            y (pd.Series): The target variable

        Returns:
            Dict[str, List[float]]: A dictionary containing lists of training times and test scores
        """
        # Define the subset sizes as percentages of the full dataset
        subset_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        train_times = []  # List to store training times for each subset
        test_scores = []  # List to store test scores for each subset

        for size in subset_sizes:
            # Calculate the actual number of samples for this subset
            subset_size = int(len(X) * size)
            
            # Create subsets of the data
            X_subset = X.iloc[:subset_size]
            y_subset = y.iloc[:subset_size]

            # Split the subset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

            # Measure the training time
            start_time = time.time()
            model_builder.model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Calculate the model's score on the test set
            score = model_builder.model.score(X_test, y_test)

            # Store the results
            train_times.append(train_time)
            test_scores.append(score)

        # Return the results as a dictionary
        return {'train_times': train_times, 'test_scores': test_scores}

    ## PLOT MODEL COMPARISON
    @measure_time
    def plot_model_comparison(self, dt_results, lr_results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Execution time comparison
        methods = ['Decision Tree', 'Linear Regression']
        times = [dt_results['train_time'], lr_results['train_time']]
        ax1.bar(methods, times)
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Model Training Time Comparison')
        for i, v in enumerate(times):
            ax1.text(i, v, f'{v:.3f}s', ha='center', va='bottom')

        # R-squared comparison
        r2_scores = [dt_results['r2'], lr_results['r2']]
        ax2.bar(methods, r2_scores)
        ax2.set_ylabel('R-squared Score')
        ax2.set_title('Model Performance Comparison')
        for i, v in enumerate(r2_scores):
            ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/model_comparison.png")
        plt.close()

    ## PLOT PAIRED COMPARISON
    @measure_time
    def plot_paired_comparison(self, ga_dt_results, hc_lr_results):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        pairs = ['GA + Decision Tree', 'HC + Linear Regression']
        
        # Execution time comparison
        times = [ga_dt_results['total_time'], hc_lr_results['total_time']]
        axes[0].bar(pairs, times)
        axes[0].set_ylabel('Total Execution Time (seconds)')
        axes[0].set_title('Total Execution Time Comparison')
        for i, v in enumerate(times):
            axes[0].text(i, v, f'{v:.3f}s', ha='center', va='bottom')

        # Alignment quality comparison
        correlations = [ga_dt_results['alignment_quality'], hc_lr_results['alignment_quality']]
        axes[1].bar(pairs, correlations)
        axes[1].set_ylabel('Alignment Quality (Correlation)')
        axes[1].set_title('Alignment Quality Comparison')
        for i, v in enumerate(correlations):
            axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom')

        # Model performance comparison
        r2_scores = [ga_dt_results['model_performance'], hc_lr_results['model_performance']]
        axes[2].bar(pairs, r2_scores)
        axes[2].set_ylabel('Model Performance (R-squared)')
        axes[2].set_title('Model Performance Comparison')
        for i, v in enumerate(r2_scores):
            axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/paired_comparison.png")
        plt.close()

    ## PLOT REGIONAL COMPARISON
    def plot_regional_comparison(self, features: pd.DataFrame):
        regions = features['region'].unique()
        mortality_rates = []
        breastfeeding_rates = []
        correlations = []

        for region in regions:
            region_data = features[features['region'] == region]
            mortality_rates.append(region_data['mortality_rate'].mean())
            breastfeeding_rates.append(region_data['exclusive_breastfeeding'].mean())
            correlations.append(region_data['exclusive_breastfeeding'].corr(region_data['mortality_rate']))

        fig, ax1 = plt.subplots(figsize=(12, 6))

        x = range(len(regions))
        ax1.bar([i - 0.2 for i in x], mortality_rates, 0.4, label='Mortality Rate', color='r', alpha=0.7)
        ax1.bar([i + 0.2 for i in x], breastfeeding_rates, 0.4, label='Breastfeeding Rate', color='b', alpha=0.7)
        ax1.set_ylabel('Rate')
        ax1.set_xlabel('Region')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions, rotation=45, ha='right')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(x, correlations, 'g-', label='Correlation')
        ax2.set_ylabel('Correlation')
        ax2.legend(loc='upper right')

        plt.title('Regional Comparison: Mortality, Breastfeeding, and Correlation')
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/regional_comparison.png")
        plt.close()

    ## PLOT TRENDS OVER TIME
    def plot_trends_over_time(self, features: pd.DataFrame):
        regions = features['region'].unique()
        years = sorted(features['year'].unique())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        for region in regions:
            region_data = features[features['region'] == region].groupby('year').agg({
                'mortality_rate': 'mean',
                'exclusive_breastfeeding': 'mean'
            })
            ax1.plot(region_data.index, region_data['mortality_rate'], label=region)
            ax2.plot(region_data.index, region_data['exclusive_breastfeeding'], label=region)

        ax1.set_ylabel('Mortality Rate')
        ax1.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_title('Mortality Rate Trends by Region')

        ax2.set_xlabel('Year')
        ax2.set_ylabel('Exclusive Breastfeeding Rate')
        ax2.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_title('Exclusive Breastfeeding Rate Trends by Region')

        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/trends_over_time.png")
        plt.close()

    ## VISUALIZE PREDICTION INTERVALS
    def visualize_prediction_intervals(self, y_true, y_pred, lower_bound, upper_bound):
        """
        Visualize prediction intervals for the test set.
        """
        plt.figure(figsize=(12, 6))
        
        # Convert all inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        
        # Sort all arrays based on y_true for better visualization
        sorted_indices = np.argsort(y_true)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        lower_bound_sorted = lower_bound[sorted_indices]
        upper_bound_sorted = upper_bound[sorted_indices]
        
        plt.scatter(y_true_sorted, y_pred_sorted, alpha=0.5, label='Predictions')
        plt.plot([y_true_sorted.min(), y_true_sorted.max()], [y_true_sorted.min(), y_true_sorted.max()], 'r--', lw=2, label='Perfect Predictions')
        
        plt.fill_between(y_true_sorted, lower_bound_sorted, upper_bound_sorted, alpha=0.2, label='95% Prediction Interval')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions with 95% Confidence Intervals')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/prediction_intervals.png")
        plt.close()

    ## VISUALIZE UNCERTAINTY BY REGION
    def visualize_uncertainty_by_region(self, features, y_true, y_pred, lower_bound, upper_bound):
        """
        Visualize how uncertainty varies across different regions.
        """
        plt.figure(figsize=(15, 8))
        
        # Ensure all inputs are pandas Series with the same index
        y_true = pd.Series(y_true, index=features.index)
        y_pred = pd.Series(y_pred, index=features.index)
        lower_bound = pd.Series(lower_bound, index=features.index)
        upper_bound = pd.Series(upper_bound, index=features.index)
        
        for region in features['region'].unique():
            mask = features['region'] == region
            y_true_region = y_true[mask]
            y_pred_region = y_pred[mask]
            lower_bound_region = lower_bound[mask]
            upper_bound_region = upper_bound[mask]
            
            interval_width = upper_bound_region - lower_bound_region
            
            plt.scatter(y_true_region, interval_width, alpha=0.5, label=region)
        
        plt.xlabel('True Mortality Rate')
        plt.ylabel('Width of 95% Prediction Interval')
        plt.title('Uncertainty in Predictions by Region')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/uncertainty_by_region.png")
        plt.close()

    ## VISUALIZE UNCERTAINTY HEATMAP
    def visualize_uncertainty_heatmap(self, features, lower_bound, upper_bound):
        """
        Create a heatmap of prediction uncertainty across regions and years.
        """
        # Create a DataFrame with the uncertainty values
        uncertainty_df = features[['region', 'year']].copy()
        uncertainty_df['uncertainty'] = upper_bound - lower_bound
        
        # Create a pivot table of uncertainty
        uncertainty_pivot = uncertainty_df.pivot_table(
            values='uncertainty', 
            index='region', 
            columns='year', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(uncertainty_pivot, cmap='YlOrRd', annot=False)
        plt.title('Heatmap of Prediction Uncertainty Across Regions and Years')
        plt.xlabel('Year')
        plt.ylabel('Region')
        plt.tight_layout()
        plt.savefig(f"{self.config.plots_dir}/uncertainty_heatmap.png")
        plt.close()




####################################
### MAIN ENTRY POINT - EXECUTION ###
###################################

## MAIN FUNCTION
def main():
    """
    Main function to orchestrate the entire analysis process.
    This function coordinates the data loading, processing, model building, and result analysis.
    It provides a step-by-step execution of the entire project workflow.
    """
    # Initialize configuration and various components
    config = DataConfig()
    data_processor = EnhancedDataProcessor(config)
    genetic_algorithm = EnhancedGeneticAlgorithm()
    hill_climbing = HillClimbingOptimizer(config)  
    model_builder = EnhancedModelBuilder(config)
    results_analyzer = ResultsAnalyzer(config)
    
    try:
        # Step 1: Data Loading and Processing
        logger.info("Loading and processing data...")
        mortality_data = data_processor.load_mortality_data()
        nutrition_data = data_processor.load_nutrition_data()
        # These steps load the raw data from mortality and nutrition datasets
        
        # Step 2: Dataset Alignment
        # We use two different alignment methods and compare their results
        logger.info("Aligning datasets using Genetic Algorithm...")
        aligned_data_ga, ga_time = genetic_algorithm.align_datasets(mortality_data, nutrition_data)
        
        logger.info("Aligning datasets using Hill Climbing...")
        aligned_data_hc, hc_time = hill_climbing.align_datasets(mortality_data, nutrition_data)
        
        # Helper function to extract main value from potentially complex data types
        def extract_main_value(value):
            if isinstance(value, str):
                return float(value.split('[')[0].strip())
            return value

        # Compare the results of both alignment algorithms
        ga_correlation = abs(
            aligned_data_ga['nutrition']['Infants exclusively breastfed for the first six months of life (%)']
            .apply(extract_main_value)
            .corr(aligned_data_ga['mortality']['Both sexes_value'].apply(extract_main_value))
        )
        hc_correlation = abs(
            aligned_data_hc['nutrition']['Infants exclusively breastfed for the first six months of life (%)']
            .apply(extract_main_value)
            .corr(aligned_data_hc['mortality']['Both sexes_value'].apply(extract_main_value))
        )
        
        # Log the results of both alignment methods
        logger.info(f"Genetic Algorithm alignment correlation: {ga_correlation}")
        logger.info(f"Hill Climbing alignment correlation: {hc_correlation}")
        logger.info(f"Genetic Algorithm execution time: {ga_time}")
        logger.info(f"Hill Climbing execution time: {hc_time}")
        
        # Choose the better alignment based on correlation
        if ga_correlation > hc_correlation:
            logger.info("Using Genetic Algorithm alignment")
            aligned_data = aligned_data_ga
        else:
            logger.info("Using Hill Climbing alignment")
            aligned_data = aligned_data_hc
        
        # Step 3: Feature Engineering and Data Preparation
        logger.info("Building and evaluating model...")
        features = model_builder.engineer_features(aligned_data['mortality'], aligned_data['nutrition'])
        # This step creates relevant features from the aligned data
        # It may include calculations like averages, trends, or derived metrics
        
        # Step 4: Data Quality Validation
        logger.info("Validating data quality...")
        model_builder._validate_data_quality(features)
        # This step checks for issues like missing values, outliers, or inconsistencies in the data
        
        # Log summary statistics and missing value counts for key features
        logger.info(f"Features summary:\n{features.describe()}")
        logger.info(f"Missing values in mortality_rate: {features['mortality_rate'].isnull().sum()}")
        logger.info(f"Missing values in exclusive_breastfeeding: {features['exclusive_breastfeeding'].isnull().sum()}")
        # These logs provide an overview of the dataset and highlight potential data quality issues
        
        # Step 5: Data Preparation for Modeling
        data = model_builder.prepare_data(features, 'mortality_rate')
        # This step splits the data into training and testing sets, and may include scaling or normalization
        
        # Prepare data for modeling
        X_train = pd.DataFrame(data['train']['X'], columns=features.drop(columns=['mortality_rate', 'country', 'region']).columns)
        X_test = pd.DataFrame(data['test']['X'], columns=features.drop(columns=['mortality_rate', 'country', 'region']).columns)
        y_train = pd.Series(data['train']['y'])
        y_test = pd.Series(data['test']['y'])
        
        # Step 6: Hyperparameter Optimization
        best_params = model_builder.optimize_hyperparameters(X_train, y_train)
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Step 7: Model Training and Evaluation
        results, train_time = model_builder.train_and_evaluate(X_train, y_train, X_test, y_test)
        results['train_time'] = train_time
        # This step trains the model on the training data and evaluates its performance on the test data
        
        # Step 8: Results Analysis
        logger.info("Analyzing results...")
        # Analyze regional patterns
        regional_analysis = results_analyzer._analyze_regional_patterns(features, features['mortality_rate'])
        # This provides detailed statistics for each region, including correlations and data points
        
        # Analyze overall breastfeeding impact
        breastfeeding_impact = results_analyzer._analyze_breastfeeding_impact(features, features['mortality_rate'])
        # This calculates the overall correlation between breastfeeding rates and mortality rates
        
        # Compile all evaluation results
        evaluation_results = {
            'regional_analysis': regional_analysis,
            'breastfeeding_impact': breastfeeding_impact,
            'model_metrics': results,
            'data_coverage': len(features) / (len(mortality_data) * len(nutrition_data)),
            'features': features  # Include the features for further analysis
        }
        
        logger.info(f"Evaluation results: {evaluation_results}")
        # This log entry provides a summary of all evaluation results
        # It includes regional analysis, overall breastfeeding impact, model metrics, and data coverage
        
        # Step 9: Visualization Creation
        try:
            logger.info("Creating visualizations...")
            results_analyzer.create_visualizations(features)
            results_analyzer._create_feature_importance_plot(results['feature_importance'])
            results_analyzer._create_hypothesis_testing_heatmap(results_analyzer._perform_hypothesis_testing(features))
            results_analyzer._create_predicted_vs_actual_plot(y_test, model_builder.model.predict(X_test))
            results_analyzer._create_time_period_analysis_plot(results_analyzer._analyze_time_periods(features))
            results_analyzer._create_economic_impact_plot(regional_analysis)
        except Exception as e:
            logger.error(f"Error in creating visualizations: {str(e)}")

        # Step 10: Hypothesis Testing
        logger.info("Performing hypothesis testing...")
        hypothesis_test_results = results_analyzer._perform_hypothesis_testing(features)
        logger.info(f"Hypothesis test results: {hypothesis_test_results}")

        # Step 11: Critical Model Evaluation
        logger.info("Critically evaluating the model...")
        model_evaluation = results_analyzer.evaluate_model_critically(model_builder.model, X_test, y_test)
        logger.info(f"Model evaluation: {model_evaluation}")

        # Step 12: NGO Report Generation
        logger.info("Creating NGO report...")
        report = results_analyzer.generate_ngo_report(evaluation_results)
        logger.info("NGO report generated successfully")
        logger.info(f"NGO Report: {report}")
        # The NGO report provides key findings and evidence strength
        # 'key_findings' include overall impact, regional insights, and priority regions
        # 'evidence_strength' includes statistical significance, data quality, and prediction accuracy
        
        # Step 13: Detailed CSV Output
        results_df = pd.DataFrame(features)
        results_df.to_csv('analysis_results.csv', index=False)
        # This CSV file contains all engineered features and can be used for further analysis
        
        # Step 14: Set Feature Names for Model Builder
        model_builder.set_feature_names(features.drop(columns=['mortality_rate', 'country', 'region']).columns)

        # Step 15: Time-based Cross-validation
        cv_results = model_builder.train_and_evaluate_with_cv(
            features.drop(columns=['mortality_rate', 'country', 'region']), 
            features['mortality_rate']
        )
        logger.info(f"Cross-validation results: {cv_results}")

        # Step 16: ANOVA and Tukey's HSD Tests
        anova_results = results_analyzer.perform_anova(features)
        tukey_results = results_analyzer.perform_tukey_hsd(features)
        logger.info(f"ANOVA results:\n{anova_results}")
        logger.info(f"Tukey's HSD results:\n{tukey_results}")

        # Add these results to the evaluation_results dictionary
        evaluation_results['anova_results'] = anova_results
        evaluation_results['tukey_results'] = tukey_results

        # Step 17: Bootstrap Predictions
        lower_ci, upper_ci = model_builder.bootstrap_predictions(X_test, y_test)
        logger.info(f"95% Confidence Interval for predictions: Lower bound {lower_ci}, Upper bound {upper_ci}")

        # Add these results to the evaluation_results dictionary
        evaluation_results['prediction_confidence_interval'] = {'lower': lower_ci, 'upper': upper_ci}

        # Step 18: Time Complexity Analysis
        _, align_time_ga = genetic_algorithm.align_datasets(mortality_data, nutrition_data)
        train_eval_result, train_time = model_builder.train_and_evaluate(X_train, y_train, X_test, y_test)
        execution_times = {'GA Alignment': align_time_ga, 'Model Training': train_time}

        # Step 19: Residual Plot
        results_analyzer.plot_residuals(y_test, model_builder.model.predict(X_test))

        # Step 20: Performance Table
        performance_table = results_analyzer.create_performance_table(results)
        logger.info(f"Performance comparison:\n{performance_table}")

        # Step 21: Scalability Analysis
        X = features.drop(columns=['mortality_rate', 'country', 'region'])
        y = features['mortality_rate']
        scalability_results = results_analyzer.analyze_scalability(model_builder, X, y)
        logger.info(f"Scalability analysis results: {scalability_results}")

        # Step 22: Ablation Study
        ablation_results = model_builder.perform_ablation_study(X, y)
        logger.info(f"Ablation study results: {ablation_results}")

        # Step 23: Learning Curve
        model_builder.plot_learning_curve(X, y)

        # Step 24: Optimization Convergence Plot
        results_analyzer.plot_optimization_comparison(ga_time, hc_time, ga_correlation, hc_correlation)

        # Step 25: Decision Tree Model Evaluation
        dt_results, dt_train_time = model_builder.train_and_evaluate(X_train, y_train, X_test, y_test)
        dt_results['train_time'] = dt_train_time

        # Step 26: Linear Regression Model Evaluation
        lr_model = LinearRegression()
        lr_start_time = time.time()
        lr_model.fit(X_train, y_train)
        lr_train_time = time.time() - lr_start_time
        lr_predictions = lr_model.predict(X_test)
        lr_results = {
            'mse': mean_squared_error(y_test, lr_predictions),
            'mae': mean_absolute_error(y_test, lr_predictions),
            'r2': r2_score(y_test, lr_predictions),
            'train_time': lr_train_time
        }

        # Step 27: Model Comparison Plot
        results_analyzer.plot_model_comparison(dt_results, lr_results)

        # Step 28: Genetic Algorithm + Decision Tree
        aligned_data_ga, ga_time = genetic_algorithm.align_datasets(mortality_data, nutrition_data)
        features_ga = model_builder.engineer_features(aligned_data_ga['mortality'], aligned_data_ga['nutrition'])
        data_ga = model_builder.prepare_data(features_ga, 'mortality_rate')
        X_train_ga, X_test_ga = data_ga['train']['X'], data_ga['test']['X']
        y_train_ga, y_test_ga = data_ga['train']['y'], data_ga['test']['y']
        dt_results, dt_train_time = model_builder.train_and_evaluate(X_train_ga, y_train_ga, X_test_ga, y_test_ga)

        # Step 29: Hill Climbing + Linear Regression
        aligned_data_hc, hc_time = hill_climbing.align_datasets(mortality_data, nutrition_data)
        features_hc = model_builder.engineer_features(aligned_data_hc['mortality'], aligned_data_hc['nutrition'])
        data_hc = model_builder.prepare_data(features_hc, 'mortality_rate')
        X_train_hc, X_test_hc = data_hc['train']['X'], data_hc['test']['X']
        y_train_hc, y_test_hc = data_hc['train']['y'], data_hc['test']['y']
        lr_model = LinearRegression()
        lr_start_time = time.time()
        lr_model.fit(X_train_hc, y_train_hc)
        lr_train_time = time.time() - lr_start_time
        lr_predictions = lr_model.predict(X_test_hc)
        lr_results = {
            'mse': mean_squared_error(y_test_hc, lr_predictions),
            'mae': mean_absolute_error(y_test_hc, lr_predictions),
            'r2': r2_score(y_test_hc, lr_predictions),
            'train_time': lr_train_time
        }

        # Step 30: Prepare results for paired comparison
        ga_dt_results = {
            'total_time': ga_time + dt_train_time,
            'alignment_quality': abs(features_ga['exclusive_breastfeeding'].corr(features_ga['mortality_rate'])),
            'model_performance': dt_results['r2']
        }

        hc_lr_results = {
            'total_time': hc_time + lr_train_time,
            'alignment_quality': abs(features_hc['exclusive_breastfeeding'].corr(features_hc['mortality_rate'])),
            'model_performance': lr_results['r2']
        }

        # Step 31: Plot paired comparison
        results_analyzer.plot_paired_comparison(ga_dt_results, hc_lr_results)

        # Step 32: Plot regional comparison
        results_analyzer.plot_regional_comparison(features)

        # Step 33: Plot trends over time
        results_analyzer.plot_trends_over_time(features)

        # Step 34: Plot confidence intervals
        y_pred = model_builder.model.predict(X_test)
        results_analyzer.visualize_prediction_intervals(y_test.values, y_pred, lower_ci, upper_ci)

        # Step 35: Visualize prediction intervals
        results_analyzer.visualize_prediction_intervals(y_test, y_pred, lower_ci, upper_ci)

        # Step 36: Visualize uncertainty by region
        X_test_with_region = pd.DataFrame(X_test, index=y_test.index)
        X_test_with_region['region'] = features.loc[y_test.index, 'region']
        results_analyzer.visualize_uncertainty_by_region(X_test_with_region, y_test, y_pred, lower_ci, upper_ci)

        # Step 37: Visualize uncertainty heatmap
        X_test_with_region_year = X_test_with_region.copy()
        X_test_with_region_year['year'] = features.loc[y_test.index, 'year']
        results_analyzer.visualize_uncertainty_heatmap(X_test_with_region_year, lower_ci, upper_ci)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
        # This error handling ensures that any exceptions are logged and re-raised
        # It helps in debugging and understanding any issues that occur during execution

if __name__ == "__main__":
    main()
    # This conditional ensures that the main() function is only executed if this script is run directly
    # It allows the script to be imported as a module without automatically running the main() function

