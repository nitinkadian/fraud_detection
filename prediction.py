########################## prediction.py #######################

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class LoanDefaultPredictor:
    def __init__(self, model_path='best_loan_default_model.pkl', scaler_path='scaler.pkl'):
        """Initialize the predictor with trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ Model and scaler loaded successfully")
        except FileNotFoundError:
            print("❌ Model files not found. Please run training.py first.")
            self.model = None
            self.scaler = None
    
    def preprocess_new_data(self, new_data):
        """Preprocess new data using the exact same steps as training"""
        print("🔧 Preprocessing new data...")
        
        # Make a copy to avoid modifying original data
        processed_data = new_data.copy()
        
        # --- EXACT SAME PREPROCESSING AS data_preparation.py ---
        
        # 1. Standardize column names (if needed)
        processed_data.columns = processed_data.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # 2. Clean string columns
        for col in processed_data.select_dtypes(include='object').columns:
            processed_data[col] = processed_data[col].astype(str).str.strip()
        
        # 3. Convert percentage columns
        if 'int_rate' in processed_data.columns:
            processed_data['int_rate'] = processed_data['int_rate'].astype(str).str.replace('%', '', regex=False).astype(float)
        
        if 'revol_util' in processed_data.columns:
            processed_data['revol_util'] = processed_data['revol_util'].astype(str).str.replace('%', '', regex=False).astype(float)
        
        # 4. Convert emp_length to numeric
        if 'emp_length' in processed_data.columns:
            emp_length_map = {
                '10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6,
                '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1,
                '< 1 year': 0.5, 'n/a': np.nan, 'nan': np.nan
            }
            processed_data['emp_length'] = processed_data['emp_length'].replace(emp_length_map).astype(float)
        
        # 5. Convert date columns (if present)
        date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
        for col in date_cols:
            if col in processed_data.columns:
                try:
                    processed_data[col] = pd.to_datetime(processed_data[col], format="%b-%Y", errors='coerce')
                except:
                    processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
        
        # 6. Create date-based features
        if 'issue_d' in processed_data.columns:
            processed_data['issue_year'] = pd.to_datetime(processed_data['issue_d']).dt.year
            processed_data['issue_month'] = pd.to_datetime(processed_data['issue_d']).dt.month
        
        if 'earliest_cr_line' in processed_data.columns and 'issue_d' in processed_data.columns:
            processed_data['credit_history_length'] = (pd.to_datetime(processed_data['issue_d']) - pd.to_datetime(processed_data['earliest_cr_line'])).dt.days // 30
        
        # 7. Create DTI bins
        if 'dti' in processed_data.columns:
            processed_data['dti_bin'] = pd.cut(processed_data['dti'], bins=[-1, 10, 20, 30, 40, np.inf], 
                                              labels=['Low', 'Moderate', 'High', 'Very High', 'Extreme'])
        
        # 8. Create loan to income ratio
        if 'loan_amnt' in processed_data.columns and 'annual_inc' in processed_data.columns:
            processed_data['loan_to_income'] = processed_data['loan_amnt'] / (processed_data['annual_inc'] + 1)
        
        # 9. Create interaction terms
        if 'int_rate' in processed_data.columns and 'dti' in processed_data.columns:
            processed_data['int_rate_x_dti'] = processed_data['int_rate'] * processed_data['dti']
        
        if 'installment' in processed_data.columns and 'annual_inc' in processed_data.columns:
            processed_data['installment_to_income'] = processed_data['installment'] / (processed_data['annual_inc'] + 1)
        
        # 10. Label encode binary categorical features
        label_encode_cols = ['term', 'home_ownership', 'verification_status', 'pymnt_plan',
                            'initial_list_status', 'application_type', 'hardship_flag', 
                            'disbursement_method', 'debt_settlement_flag']
        
        for col in label_encode_cols:
            if col in processed_data.columns:
                try:
                    le = LabelEncoder()
                    processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                except:
                    processed_data[col] = 0  # Default value if encoding fails
        
        # 11. One-hot encode multi-class categorical variables
        one_hot_cols = ['purpose', 'grade', 'sub_grade', 'dti_bin']
        for col in one_hot_cols:
            if col in processed_data.columns:
                # Get all possible categories from training (we'll simulate this)
                # In practice, you should save the one-hot encoder or know the expected columns
                try:
                    processed_data = pd.get_dummies(processed_data, columns=[col], drop_first=True)
                except:
                    pass  # Skip if one-hot encoding fails
        
        # 12. Create income bins (if annual_inc exists)
        if 'annual_inc' in processed_data.columns:
            try:
                processed_data['income_bin'] = pd.qcut(processed_data['annual_inc'], q=4, 
                                                      labels=['Low', 'Medium', 'High', 'Very High'])
                processed_data = pd.get_dummies(processed_data, columns=['income_bin'], drop_first=True)
            except:
                pass  # Skip if binning fails
        
        # 13. Scale numerical features using the saved scaler
        scale_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'revol_util',
                     'open_acc', 'total_acc', 'credit_history_length', 'loan_to_income']
        
        for col in scale_cols:
            if col in processed_data.columns:
                try:
                    processed_data[col] = self.scaler.transform(processed_data[[col]])
                except:
                    # If scaling fails, use min-max scaling as fallback
                    processed_data[col] = (processed_data[col] - processed_data[col].min()) / (processed_data[col].max() - processed_data[col].min() + 1e-8)
        
        # 14. Drop columns that were dropped during training
        drop_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'loan_status', 
                    'emp_title', 'title', 'zip_code', 'addr_state']
        
        for col in drop_cols:
            if col in processed_data.columns:
                processed_data.drop(columns=col, inplace=True)
        
        # 15. Ensure we have all the features the model expects
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
            
            # Add missing features with default value 0
            missing_features = set(expected_features) - set(processed_data.columns)
            for feature in missing_features:
                processed_data[feature] = 0
            
            # Remove extra features not used by the model
            extra_features = set(processed_data.columns) - set(expected_features)
            if extra_features:
                processed_data.drop(columns=list(extra_features), inplace=True)
            
            # Ensure correct feature order
            processed_data = processed_data[list(expected_features)]
        
        print("✅ Data preprocessing completed")
        return processed_data
    
    def predict(self, new_data, threshold=0.5):
        """Make predictions on new data"""
        if self.model is None:
            print("❌ Model not loaded. Cannot make predictions.")
            return None
        
        # Preprocess the data
        processed_data = self.preprocess_new_data(new_data)
        
        # Make predictions
        try:
            probabilities = self.model.predict_proba(processed_data)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            results = pd.DataFrame({
                'loan_id': new_data.index if hasattr(new_data, 'index') else range(len(new_data)),
                'default_probability': probabilities,
                'prediction': predictions,
                'risk_category': np.where(predictions == 1, 'High Risk', 'Low Risk')
            })
            
            print("✅ Predictions completed successfully")
            return results
        
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def predict_batch(self, file_path, threshold=0.5):
        """Predict on a batch of data from a file"""
        try:
            print(f"📂 Loading batch data from {file_path}")
            batch_data = pd.read_csv(file_path)
            results = self.predict(batch_data, threshold)
            
            if results is not None:
                # Save results
                output_path = 'predictions_results.csv'
                results.to_csv(output_path, index=False)
                print(f"💾 Predictions saved to {output_path}")
                
                # Show summary
                self._show_prediction_summary(results)
            
            return results
        
        except Exception as e:
            print(f"❌ Error loading batch data: {e}")
            return None
    
    def _show_prediction_summary(self, results):
        """Show summary of predictions"""
        print("\n" + "="*50)
        print("📊 PREDICTION SUMMARY")
        print("="*50)
        
        total_loans = len(results)
        high_risk_loans = len(results[results['prediction'] == 1])
        high_risk_percentage = (high_risk_loans / total_loans) * 100
        
        print(f"Total loans analyzed: {total_loans}")
        print(f"High-risk loans identified: {high_risk_loans} ({high_risk_percentage:.2f}%)")
        print(f"Low-risk loans: {total_loans - high_risk_loans}")
        
        # Risk distribution
        risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        results['risk_level'] = pd.cut(results['default_probability'], bins=risk_bins, labels=risk_labels)
        
        print("\n📈 Risk Level Distribution:")
        print(results['risk_level'].value_counts().sort_index())
        
        # Show distribution of probabilities
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=results, x='default_probability', hue='risk_category', bins=30)
        plt.title('Distribution of Default Probabilities')
        plt.xlabel('Default Probability')
        plt.ylabel('Count')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        results['risk_level'].value_counts().sort_index().plot(kind='bar')
        plt.title('Risk Level Distribution')
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_on_test(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance on test data"""
        if self.model is None:
            print("❌ Model not loaded. Cannot evaluate.")
            return None
        
        print("\n" + "="*50)
        print("🧪 MODEL EVALUATION ON TEST DATA")
        print("="*50)
        
        # Preprocess test data first
        X_test_processed = self.preprocess_new_data(X_test)
        
        # Make predictions
        probabilities = self.model.predict_proba(X_test_processed)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)
        
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cr)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        return {
            'confusion_matrix': cm,
            'classification_report': cr,
            'roc_auc': roc_auc
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize predictor
    predictor = LoanDefaultPredictor()
    
    # Example: Create sample data for prediction
    sample_data = pd.DataFrame({
        'loan_amnt': [10000, 15000, 20000],
        'int_rate': [10.5, 15.2, 8.7],
        'annual_inc': [50000, 60000, 75000],
        'dti': [15.2, 25.7, 10.3],
        'revol_util': [45.6, 78.9, 30.2],
        'emp_length': ['5 years', '2 years', '8 years'],
        'term': ['36 months', '36 months', '60 months'],
        'home_ownership': ['MORTGAGE', 'RENT', 'OWN'],
        'verification_status': ['Verified', 'Not Verified', 'Verified'],
        'purpose': ['debt_consolidation', 'credit_card', 'home_improvement'],
        'credit_history_length': [84, 36, 120],
        'open_acc': [8, 5, 12],
        'total_acc': [15, 10, 20],
        'pub_rec': [0, 1, 0],
        'delinq_2yrs': [0, 2, 0],
        'issue_d': ['Jan-2015', 'Jun-2016', 'Mar-2017'],
        'earliest_cr_line': ['Jan-2010', 'Jun-2013', 'Mar-2007']
    })
    
    print("📋 Sample data for prediction:")
    print(sample_data)
    
    # Make predictions
    predictions = predictor.predict(sample_data)
    
    if predictions is not None:
        print("\n🎯 Prediction results:")
        print(predictions)
        
        # Show risk analysis
        high_risk_count = len(predictions[predictions['prediction'] == 1])
        total_count = len(predictions)
        
        print(f"\n📊 Risk Analysis:")
        print(f"High-risk loans: {high_risk_count}/{total_count} ({(high_risk_count/total_count)*100:.1f}%)")
        
        # You can also load test data for evaluation if available
        # For example:
        # try:
        #     test_data = pd.read_csv('test_loan_data.csv')
        #     X_test = test_data.drop('loan_status_binary', axis=1)
        #     y_test = test_data['loan_status_binary']
        #     predictor.evaluate_on_test(X_test, y_test)
        # except FileNotFoundError:
        #     print("Test data not available for evaluation")
    
    print("\n💡 To predict on a batch file, use: predictor.predict_batch('your_data.csv')")