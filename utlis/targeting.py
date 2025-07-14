import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from utlis.data_utils import get_feature_cols


def calculate_revenues(test_df: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate revenue per client from test data using regression models.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataframe containing features for revenue prediction
    models : Dict[str, Any]
        Dictionary containing trained regression models for each product
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added revenue columns: Revenue_CL, Revenue_CC, Revenue_MF
    """
    
    # Make a copy to avoid modifying original dataframe
    result_df = test_df.copy()
    
    # Calculate revenue for each product
    for product in ['CL', 'CC', 'MF']:
        revenue_col = f'Revenue_{product}'
        
        # Get the model for this product
        model = models.get(f'{product}_revenue')

        # Get feature columns using data_utils function
        feature_cols = get_feature_cols(test_df)
        
        # Predict revenue directly using regression model
        predicted_revenue = model.predict(test_df[feature_cols])
        
        # Add to result dataframe
        result_df[revenue_col] = predicted_revenue
    
    return result_df



def assign_best_offer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick the best offer for each client based on expected value.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with propensity scores (p_cl, p_cc, p_mf) and 
        predicted revenues (Revenue_CL, Revenue_CC, Revenue_MF)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added columns: Expected_Revenue_CL, Expected_Revenue_CC, 
        Expected_Revenue_MF, Best_Offer, Expected_Revenue
    """
    offers = ['CL', 'CC', 'MF']
    
    # Calculate expected values using propensity scores and predicted revenues
    expected = np.vstack([
        df['p_cl'] * df['Revenue_CL'],
        df['p_cc'] * df['Revenue_CC'], 
        df['p_mf'] * df['Revenue_MF']
    ]).T
    
    # Add individual expected revenue columns for each product
    df['Expected_Revenue_CL'] = expected[:, 0]
    df['Expected_Revenue_CC'] = expected[:, 1]
    df['Expected_Revenue_MF'] = expected[:, 2]
    
    # Find the offer with maximum expected value for each client
    best_offer_idx = np.argmax(expected, axis=1)
    
    df['Best_Offer'] = [offers[i] for i in best_offer_idx]
    df['Expected_Revenue'] = expected[np.arange(len(df)), best_offer_idx]
    
    return df
    

def select_top_targets(df: pd.DataFrame, top_frac: float = 0.15) -> pd.DataFrame:
    """
    Build the final list by ranking clients by expected revenue.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with Expected_Revenue column
    top_frac : float, default=0.15
        Fraction of top clients to select (e.g., 0.15 = top 15%)
        
    Returns:
    --------
    pd.DataFrame
        Top clients sorted by expected revenue with relevant columns
    """
    n = int(len(df) * top_frac)
    df_sorted = df.sort_values('Expected_Revenue', ascending=False).head(n)
    
    # Select relevant columns for final output
    output_cols = ['Client', 'Best_Offer', 'Expected_Revenue']
    if 'Age' in df.columns:
        output_cols.append('Age')
    if 'Tenure' in df.columns:
        output_cols.append('Tenure')
    
    return df_sorted[output_cols]


def calculate_revenue_forecast(targets_df: pd.DataFrame, df_best_offer: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate revenue forecast and lift vs baseline targeting.
    
    Parameters:
    -----------
    targets_df : pd.DataFrame
        Dataframe containing targeted clients with Expected_Revenue
    df_best_offer : pd.DataFrame
        Full dataframe with all clients and their Expected_Revenue
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing forecast metrics including total revenue, 
        number of targets, lift percentage, etc.
    """
    # Expected revenue from targeted clients
    total_expected_revenue = targets_df['Expected_Revenue'].sum()

    # Calculate average revenue 
    n_targets = len(targets_df)
    avg_baseline_revenue = df_best_offer['Expected_Revenue'].mean()
    baseline_revenue = n_targets * avg_baseline_revenue
    
    # Calculate lift
    lift = total_expected_revenue / baseline_revenue if baseline_revenue > 0 else 0
    
    forecast_results = {
        'total_expected_revenue': total_expected_revenue,
        'targeted_clients': n_targets,
        'avg_baseline_revenue_per_client': avg_baseline_revenue,
        'baseline_revenue': baseline_revenue,
        'lift_vs_baseline': lift,
        'lift_percentage': (lift - 1) * 100 if lift > 0 else 0
    }
    
    return forecast_results

def run_full_targeting_pipeline(predicted_revenues_df: pd.DataFrame, top_frac: float = 0.15) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Run complete stages 3-6 targeting pipeline.
    
    Parameters:
    -----------
    predicted_revenues_df : pd.DataFrame
        Dataframe with propensity scores and predicted revenues
    top_frac : float, default=0.15
        Fraction of top clients to target
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]
        (targeted_clients, forecast_results, full_dataframe_with_offers)
    """
    print("Stage 3,4: Assigning best offers...")
    df_best_offer = assign_best_offer(predicted_revenues_df)
    
    print("Stage 5: Selecting top targets...")
    targets = select_top_targets(df_best_offer, top_frac)
    
    print("Stage 6: Calculating revenue forecast...")
    forecast = calculate_revenue_forecast(targets, df_best_offer)
    
    return targets, forecast, df_best_offer



def print_targeting_summary(targets: pd.DataFrame, forecast: Dict[str, Any]) -> None:
    """
    Print comprehensive targeting summary.
    
    Parameters:
    -----------
    targets : pd.DataFrame
        Dataframe containing targeted clients
    forecast : Dict[str, Any]
        Dictionary containing forecast results
        
    Returns:
    --------
    None
        Prints summary to console
    """
    print("\n=== TARGETING SUMMARY ===")
    print(f"Total clients targeted: {forecast['targeted_clients']}")
    print(f"Total expected revenue: ${forecast['total_expected_revenue']:.2f}")
    print(f"Average expected revenue per client: ${forecast['avg_baseline_revenue_per_client']:.2f}")
    print(f"Lift vs baseline targeting: {forecast['lift_percentage']:.1f}%")
    
    print(f"\nOffer distribution:")
    offer_counts = targets['Best_Offer'].value_counts()
    for offer, count in offer_counts.items():
        percentage = (count / len(targets)) * 100
        print(f"  {offer}: {count} clients ({percentage:.1f}%)")