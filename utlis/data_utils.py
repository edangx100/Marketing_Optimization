import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional


def load_data(file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from Excel file with multiple sheets.
    
    Parameters:
    -----------
    file : str
        Path to the Excel file
        
    Returns:
    --------
    tuple
        Four dataframes: soc_dem, products, inflow, sales
    """
    soc_dem = pd.read_excel(file, sheet_name='Soc_Dem')
    products = pd.read_excel(file, sheet_name='Products_ActBalance')
    inflow = pd.read_excel(file, sheet_name='Inflow_Outflow')
    sales = pd.read_excel(file, sheet_name='Sales_Revenues')
    return soc_dem, products, inflow, sales


def merge_data(soc_dem: pd.DataFrame, products: pd.DataFrame, inflow: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all dataframes on Client ID.
    
    Parameters:
    -----------
    soc_dem : pandas.DataFrame
        Socio-demographic data
    products : pandas.DataFrame
        Product and account balance data
    inflow : pandas.DataFrame
        Inflow/outflow transaction data
    sales : pandas.DataFrame
        Sales and revenue data
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe with all client information
    """
    df = soc_dem.merge(products, on='Client', how='left')
    df = df.merge(inflow, on='Client', how='left')
    df = df.merge(sales, on='Client', how='left')
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Get feature columns for modeling, excluding target and ID columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of feature column names
    """
    exclude = ['Client','Sale_CL', 'Sale_CC', 'Sale_MF', 'Revenue_CL','Revenue_CC','Revenue_MF', 'p_cl', 'p_cc', 'p_mf'] 
    return [col for col in df.columns if col not in exclude]


def get_target_cols(df: pd.DataFrame) -> List[str]:
    """
    Get target columns for modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of target column names
    """
    return ['Sale_CL', 'Sale_CC', 'Sale_MF', 'Revenue_CL','Revenue_CC','Revenue_MF']


def process_features1(df: pd.DataFrame, le: Optional[LabelEncoder] = None) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Handle missing values based on feature processing requirements.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with missing values handled
    """
    df = df.copy()
    
    # Drop missing values for Sex and apply label encoding
    if 'Sex' in df.columns:
        df = df.dropna(subset=['Sex'])
        if le is None:
            le = LabelEncoder()
            df['Sex'] = le.fit_transform(df['Sex'])
        else:
            df['Sex'] = le.transform(df['Sex'])
    
    # Replace missing values with 0 for count and balance features
    count_cols = ['Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL']
    actbal_cols = ['ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL']
    
    for col in count_cols + actbal_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(0)
    
    # Replace missing values with 0 
    volume_cred_cols = ['VolumeCred', 'VolumeCred_CA']
    transactions_cred_cols = ['TransactionsCred', 'TransactionsCred_CA']
    
    for col in volume_cred_cols + transactions_cred_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(0)
    
    # Replace missing values with 0 
    volume_deb_cols = ['VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 
                       'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder']
    transactions_deb_cols = ['TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 
                            'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder']
    
    for col in volume_deb_cols + transactions_deb_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(0)
    
    return df, le


def process_features2(df: pd.DataFrame, le: Optional[LabelEncoder] = None) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Handle missing values and create additional features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    le : Optional[LabelEncoder]
        Label encoder for Sex column. If None, creates new one.
        
    Returns:
    --------
    tuple
        (processed_dataframe, label_encoder)
    """
    df = df.copy()
    
    # Drop missing values for Sex and apply label encoding
    if 'Sex' in df.columns:
        df = df.dropna(subset=['Sex'])
        if le is None:
            le = LabelEncoder()
            df['Sex'] = le.fit_transform(df['Sex'])
        else:
            df['Sex'] = le.transform(df['Sex'])
    
    # Replace missing values with 0 for count and balance features
    count_cols = ['Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL']
    actbal_cols = ['ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL']
    
    for col in count_cols + actbal_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(0)
    
    # Replace missing values with 0 for volume and transaction features
    volume_cred_cols = ['VolumeCred', 'VolumeCred_CA']
    transactions_cred_cols = ['TransactionsCred', 'TransactionsCred_CA']
    
    for col in volume_cred_cols + transactions_cred_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(0)
    
    # Replace missing values with 0 for debit features
    volume_deb_cols = ['VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 
                    'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder']
    transactions_deb_cols = ['TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 
                            'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder']
    
    for col in volume_deb_cols + transactions_deb_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(0)

    # Create new feature VolumeCredDebRatio = VolumeCred/VolumeDeb
    if 'VolumeCred' in df.columns and 'VolumeDeb' in df.columns:
        df['VolumeCredDebRatio'] = df['VolumeCred'] / (df['VolumeDeb'] + 1)  # +1 to avoid division by zero

    # Drop VolumeCred feature
    if 'VolumeCred' in df.columns:
        df = df.drop('VolumeCred', axis=1)

    # Drop VolumeDeb_CA feature
    if 'VolumeDeb_CA' in df.columns:
        df = df.drop('VolumeDeb_CA', axis=1)

    # Drop TransactionsCred_CA feature
    if 'TransactionsCred_CA' in df.columns:
        df = df.drop('TransactionsCred_CA', axis=1)

    # Drop TransactionsDeb_CA feature
    if 'TransactionsDeb_CA' in df.columns:
        df = df.drop('TransactionsDeb_CA', axis=1)

    return df, le