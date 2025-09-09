import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load processed data"""
    print("📂 Loading processed data...")
    df = pd.read_csv('processed_loan_data.csv')
    
    # Separate features and target
    X = df.drop('loan_status_binary', axis=1)
    y = df['loan_status_binary']
    
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✅ Class distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_test_split_data(X, y):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_models_with_smote(X_train, y_train, X_test, y_test):
    """Train models with SMOTE oversampling"""
    print("\n" + "="*50)
    print("🤖 TRAINING MODELS WITH SMOTE OVERSAMPLING")
    print("="*50)
    
    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"✅ After SMOTE - Class distribution: {np.bincount(y_train_res)}")
    
    models = {
        'LogisticRegression_SMOTE': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest_SMOTE': RandomForestClassifier(random_state=42),
        'XGBoost_SMOTE': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train_res, y_train_res)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        results[name] = {
            'model': model,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'f1_score': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        print(f"✅ {name} trained - F1: {results[name]['f1_score']:.4f}, Recall: {results[name]['recall']:.4f}")
    
    return results

def train_models_class_weight(X_train, y_train, X_test, y_test):
    """Train models with class weights"""
    print("\n" + "="*50)
    print("⚖️ TRAINING MODELS WITH CLASS WEIGHTS")
    print("="*50)
    
    # Calculate class weights
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    
    models = {
        'LogisticRegression_ClassWeight': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'RandomForest_ClassWeight': RandomForestClassifier(
            class_weight='balanced', random_state=42
        ),
        'XGBoost_ClassWeight': XGBClassifier(
            scale_pos_weight=scale_pos_weight, 
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results[name] = {
            'model': model,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        print(f"✅ {name} trained - F1: {results[name]['f1_score']:.4f}, Recall: {results[name]['recall']:.4f}")
    
    return results

def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Perform hyperparameter tuning with GridSearchCV"""
    print("\n" + "="*50)
    print("🎯 HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("="*50)
    
    # Define parameter grids
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', None]
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'scale_pos_weight': [1, (np.bincount(y_train)[0] / np.bincount(y_train)[1])]
        },
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'class_weight': ['balanced', None],
            'solver': ['liblinear', 'saga']
        }
    }
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        
        # Use stratified K-fold for cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=cv, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        results[f'{name}_Tuned'] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV F1: {grid_search.best_score_:.4f}")
        print(f"✅ Test F1: {results[f'{name}_Tuned']['f1_score']:.4f}")
    
    return results

def evaluate_all_models(all_results, X_test, y_test):
    """Evaluate and compare all models"""
    print("\n" + "="*50)
    print("📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    
    evaluation_df = pd.DataFrame()
    
    for approach, results in all_results.items():
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Approach': approach,
                'F1_Score': metrics['f1_score'],
                'Recall': metrics['recall'],
                'ROC_AUC': metrics['roc_auc'],
                'Precision': metrics['classification_report']['1']['precision'],
                'Support_1': metrics['classification_report']['1']['support']
            }
            evaluation_df = pd.concat([evaluation_df, pd.DataFrame([row])], ignore_index=True)
    
    # Sort by F1 score
    evaluation_df = evaluation_df.sort_values('F1_Score', ascending=False)
    
    print("🏆 Model Performance Comparison:")
    print(evaluation_df[['Model', 'Approach', 'F1_Score', 'Recall', 'ROC_AUC', 'Precision']].to_string(index=False))
    
    # Select best model based on F1 score
    best_model_row = evaluation_df.iloc[0]
    best_model_name = best_model_row['Model']
    best_approach = best_model_row['Approach']
    best_model = all_results[best_approach][best_model_name]['model']
    
    print(f"\n🎉 BEST MODEL: {best_model_name}")
    print(f"📈 F1 Score: {best_model_row['F1_Score']:.4f}")
    print(f"🔍 Recall: {best_model_row['Recall']:.4f}")
    print(f"📊 ROC AUC: {best_model_row['ROC_AUC']:.4f}")
    
    return best_model, evaluation_df

def save_best_model(best_model, evaluation_df):
    """Save the best model and evaluation results"""
    joblib.dump(best_model, 'best_loan_default_model.pkl')
    evaluation_df.to_csv('model_evaluation_results.csv', index=False)
    
    print("💾 Saved best model to 'best_loan_default_model.pkl'")
    print("💾 Saved evaluation results to 'model_evaluation_results.csv'")

def plot_performance_comparison(evaluation_df):
    """Plot model performance comparison"""
    plt.figure(figsize=(12, 8))
    
    # Create a combined label for x-axis
    evaluation_df['Model_Approach'] = evaluation_df['Model'] + ' (' + evaluation_df['Approach'] + ')'
    
    # Plot F1 scores
    plt.subplot(2, 2, 1)
    sns.barplot(x='F1_Score', y='Model_Approach', data=evaluation_df)
    plt.title('F1 Score Comparison')
    plt.xlabel('F1 Score')
    
    # Plot Recall scores
    plt.subplot(2, 2, 2)
    sns.barplot(x='Recall', y='Model_Approach', data=evaluation_df)
    plt.title('Recall Comparison')
    plt.xlabel('Recall')
    
    # Plot ROC AUC scores
    plt.subplot(2, 2, 3)
    sns.barplot(x='ROC_AUC', y='Model_Approach', data=evaluation_df)
    plt.title('ROC AUC Comparison')
    plt.xlabel('ROC AUC')
    
    # Plot Precision scores
    plt.subplot(2, 2, 4)
    sns.barplot(x='Precision', y='Model_Approach', data=evaluation_df)
    plt.title('Precision Comparison')
    plt.xlabel('Precision')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    # Train models with different approaches
    all_results = {}
    
    # 1. SMOTE approach
    smote_results = train_models_with_smote(X_train, y_train, X_test, y_test)
    all_results['SMOTE'] = smote_results
    
    # 2. Class weight approach
    class_weight_results = train_models_class_weight(X_train, y_train, X_test, y_test)
    all_results['ClassWeight'] = class_weight_results
    
    # 3. Hyperparameter tuning approach
    tuned_results = hyperparameter_tuning(X_train, y_train, X_test, y_test)
    all_results['Tuned'] = tuned_results
    
    # Evaluate and select best model
    best_model, evaluation_df = evaluate_all_models(all_results, X_test, y_test)
    
    # Plot results
    plot_performance_comparison(evaluation_df)
    
    # Save best model
    save_best_model(best_model, evaluation_df)
    
    print("\n🎯 Training completed! Best model saved and ready for prediction.")