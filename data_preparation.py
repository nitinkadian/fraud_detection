import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

def load_and_clean_data(data_path):
    """Load and clean the loan data"""
    print("📊 Loading and cleaning data...")
    
    # Load data
    df = pd.read_csv(data_path, low_memory=False)
    
    # Drop leaky features (post-approval data)
    leaky_features = ['recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'total_pymnt', 
                      'total_pymnt_inv', 'total_rec_prncp', 'total_rec_late_fee']
    df.drop(columns=[col for col in leaky_features if col in df.columns], inplace=True)
    print(f"✅ Dropped leaky features: {[col for col in leaky_features if col in df.columns]}")
    
    # Drop completely empty columns
    empty_cols = df.columns[df.isnull().all()]
    df.drop(columns=empty_cols, inplace=True)
    print(f"✅ Dropped {len(empty_cols)} completely empty columns.")
    
    # Drop duplicate rows
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"✅ Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print("✅ Standardized column names.")
    
    # Clean string columns
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip()
    print("✅ Cleaned string columns.")
    
    # Convert date columns
    date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%b-%Y", errors='coerce')
    print("✅ Converted date columns.")
    
    # Convert percentage columns
    if 'int_rate' in df.columns:
        df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '', regex=False).astype(float)
    
    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '', regex=False).astype(float)
    print("✅ Converted percentage columns to float.")
    
    # Convert emp_length to numeric
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].replace({
            '10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6,
            '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1,
            '< 1 year': 0.5, 'n/a': np.nan
        }).astype(float)
        print("✅ Cleaned 'emp_length' column.")
    
    # Impute missing values
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))
    
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
    print("✅ Imputed missing values.")
    
    # Remove unrealistic values
    if 'annual_inc' in df.columns:
        df = df[df['annual_inc'] >= 0]
    if 'loan_amnt' in df.columns:
        df = df[df['loan_amnt'] >= 0]
    print("✅ Removed rows with negative values.")
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    print("✅ Final cleaned data shape:", df.shape)
    
    return df

def create_features(df):
    """Create engineered features"""
    print("🔧 Creating features...")
    
    # Date-based features
    if 'issue_d' in df.columns:
        df['issue_year'] = pd.to_datetime(df['issue_d']).dt.year
        df['issue_month'] = pd.to_datetime(df['issue_d']).dt.month
    
    if 'earliest_cr_line' in df.columns:
        df['credit_history_length'] = (pd.to_datetime(df['issue_d']) - pd.to_datetime(df['earliest_cr_line'])).dt.days // 30
    
    if 'last_pymnt_d' in df.columns:
        df['months_since_last_payment'] = (pd.to_datetime(df['issue_d']) - pd.to_datetime(df['last_pymnt_d'])).dt.days // 30
    
    # DTI bins
    if 'dti' in df.columns:
        df['dti_bin'] = pd.cut(df['dti'], bins=[-1, 10, 20, 30, 40, np.inf], 
                              labels=['Low', 'Moderate', 'High', 'Very High', 'Extreme'])
    
    # Loan to income ratio
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    
    # Target variable
    if 'loan_status' in df.columns:
        df['loan_status_binary'] = df['loan_status'].apply(
            lambda x: 1 if x in ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
        )
    
    # Interaction terms
    df['int_rate_x_dti'] = df['int_rate'] * df['dti']
    df['installment_to_income'] = df['installment'] / (df['annual_inc'] + 1)
    print("✅ Created interaction terms.")
    
    return df

def encode_and_scale(df):
    """Encode categorical variables and scale numerical features"""
    print("🎛️ Encoding and scaling features...")
    
    # Label encode binary categorical features
    label_encode_cols = ['term', 'home_ownership', 'verification_status', 'pymnt_plan',
                         'initial_list_status', 'application_type', 'hardship_flag', 
                         'disbursement_method', 'debt_settlement_flag']
    
    for col in label_encode_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    print("✅ Label encoded binary categorical features.")
    
    # One-hot encode multi-class categorical variables
    one_hot_cols = ['purpose', 'grade', 'sub_grade', 'dti_bin']
    df = pd.get_dummies(df, columns=[col for col in one_hot_cols if col in df.columns], drop_first=True)
    print("✅ One-hot encoded categorical features.")
    
    # Bin income into groups
    if 'annual_inc' in df.columns:
        df['income_bin'] = pd.qcut(df['annual_inc'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        df = pd.get_dummies(df, columns=['income_bin'], drop_first=True)
    print("✅ Created income bins and encoded them.")
    
    # Scale numerical features
    scale_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'revol_util',
                  'open_acc', 'total_acc', 'credit_history_length', 'loan_to_income']
    
    scaler = MinMaxScaler()
    for col in scale_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
    
    # Save scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Scaled features and saved scaler.")
    
    # Drop unnecessary columns
    drop_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'loan_status', 
                 'emp_title', 'title', 'zip_code', 'addr_state']
    
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    print("✅ Dropped unnecessary columns.")
    
    return df

def prepare_model_data(df):
    """Prepare final dataset for modeling"""
    print("📈 Preparing model data...")
    
    # Select safe features for modeling
    safe_features = [
        'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'loan_to_income', 'revol_util',
        'credit_history_length', 'open_acc', 'total_acc', 'pub_rec', 'delinq_2yrs',
        'emp_length', 'term', 'home_ownership', 'verification_status',
        'installment_to_income', 'int_rate_x_dti'
    ]
    
    # Add one-hot encoded columns that start with purpose_, grade_, etc.
    for col in df.columns:
        if col.startswith(('purpose_', 'grade_', 'sub_grade_', 'dti_bin_', 'income_bin_')):
            safe_features.append(col)
    
    target = 'loan_status_binary'
    
    # Keep only relevant columns
    model_columns = [col for col in safe_features if col in df.columns] + [target]
    df_model = df[model_columns].dropna()
    
    print(f"✅ Final model dataset shape: {df_model.shape}")
    print(f"✅ Features: {len(model_columns) - 1}, Target: {target}")
    
    return df_model

if __name__ == "__main__":
    # Example usage
    data_path = r"C:\Users\IPE-1970\Downloads\Project for\loan.csv"
    df = load_and_clean_data(data_path)
    df = create_features(df)
    df = encode_and_scale(df)
    df_model = prepare_model_data(df)
    
    # Save processed data
    df_model.to_csv('processed_loan_data.csv', index=False)
    print("💾 Saved processed data to 'processed_loan_data.csv'")