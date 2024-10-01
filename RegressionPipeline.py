from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class RegressionPipeline:
    def __init__(self, df, target_col, test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.features_train = None
        self.features_test = None
        self.target_train = None
        self.target_test = None
        self.result = None
        self.scaler = StandardScaler()

        self.run()

    def split_data(self):
        """
        Splits the DataFrame into training and testing sets.
        """
        features = self.df.drop(columns=[self.target_col])
        target = self.df[self.target_col]
        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(
            features, target, test_size=self.test_size, random_state=self.random_state)

    def standardize_data(self):
        """
        Standardizes the features using StandardScaler.
        """
        self.features_train = self.scaler.fit_transform(self.features_train)
        self.features_test = self.scaler.transform(self.features_test)

    def train_and_evaluate_models(self):
        """
        Trains and evaluates multiple regression models.
        """
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Lasso Regression': Lasso(alpha=0.1),
            'Ridge Regression': Ridge(alpha=1.0),
            'ElasticNet Regression': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        results = {}
        for model_name, model_instance in regression_models.items():
            model_instance.fit(self.features_train, self.target_train)
            target_pred = model_instance.predict(self.features_test)
            results[model_name] = {
                'R-squared': r2_score(self.target_test, target_pred),
                'RMSE': np.sqrt(mean_squared_error(self.target_test, target_pred))
            }
        
        return results, regression_models['Lasso Regression']

    def display_results(self, results):
        """
        Displays the R-squared and RMSE for each regression model.
        """
        for model_name, metrics in results.items():
            print(f'{model_name}:')
            print(f'  R-squared: {metrics["R-squared"]}')
            print(f'  RMSE: {metrics["RMSE"]}\n')

    def display_feature_importance(self, model):
        """
        Displays the feature importance (coefficients) for the given model.
        """
        print("Lasso Coefficients:")
        for feature, coefficient in zip(self.df.drop(columns=[self.target_col]).columns, model.coef_):
            print(f"{feature}: {coefficient}")

    def run(self):
        """
        Complete regression pipeline function.
        """
        # Step 1: Split the data
        self.split_data()
        
        # Step 2: Standardize the data
        self.standardize_data()
        
        # Step 3: Train and evaluate models
        results, lasso_model = self.train_and_evaluate_models()
        
        # Step 4: Display results
        self.display_results(results)

        # Step 5: Display Lasso feature importance
        self.display_feature_importance(lasso_model)

        self.result = results
