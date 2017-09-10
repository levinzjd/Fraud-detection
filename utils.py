import pandas as pd


# Labels for flagging events considered fraud
fraud_list = ['fraudster', 'fraudster_att', 'fraudster_event']

# g1 are channels with higher probability of fraud
# g2 are channels with lower probability of fraud
channel_dict = {'g1': [0,7], 'g2': [4,8, 10, 12]}

# Features we engineered from our data to use in our model
cols = ['venue_name_exist', 'is_new_user','channel_g1', \
      'channel_g2', 'num_payouts_above_30', 'delivery_method_iszero', \
      'whether_previous_payouts', 'name_is_capped']

def preprocess_series(df):
    """
    Preprocessing if the dataframe only has one row
    Return design matrix
    """

    df['is_new_user'] = int(df['user_age'] == 0)
    df['venue_name_exist'] = int(df['venue_name']=='')
    #df['sum_ticket_sold'] = df['ticket_types'].apply(lambda x: sum_ticket_sold(x))
    df['num_payouts_above_30'] = int(df['num_payouts'] > 30)
    df['delivery_method_iszero'] = int(df['delivery_method'] == 0)
    df['channel_g1'] = int(df['channels'] in channel_dict['g1'])
    df['channel_g2'] = int(df['channels'] in channel_dict['g2'])
    df['whether_previous_payouts'] = int(len(df['previous_payouts']) > 0)
    df['name_is_capped'] = int((df['org_name'].lower() == df['org_name']))
    X = df[cols].values
    return X

def preprocess(df, label=True):
    """
    Preprocessing and Feature Engineering
    Return design matrix
    """

    # Engineered features
    df['is_new_user'] = (df['user_age'] == 0).astype(int)
    df['venue_name_exist'] = (df['venue_name'].isnull()).astype(int)
    df['num_payouts_above_30'] = (df['num_payouts']> 30).astype(int)
    df['delivery_method_iszero'] = (df['delivery_method'] == 0).astype(int)
    df['channel_g1'] = (df['channels'].apply(lambda x: x in channel_dict['g1'])).astype(int)
    df['channel_g2'] = (df['channels'].apply(lambda x: x in channel_dict['g2'])).astype(int)
    df['whether_previous_payouts'] = (df['previous_payouts'].apply(lambda x: len(x) > 0)).astype(int)
    df['name_is_capped'] = (df['org_name'].apply(lambda x: x.lower() == x)).astype(int)
    X = df[cols].values

    # Labels for fraud
    if label:
        df['fraud'] = (df['acct_type'].apply(lambda x: x in fraud_list)).astype(int)
        y = df['fraud'].values
        return X, y
    return X

# Gets the most important features from our RF model
def get_import_features(rf, n=3):
    tuples = sorted((zip(rf.feature_importances_, cols)), reverse=True)
    return tuples[:n]

# Creates the cost matrix for our model
def build_cost_matrix(y, y_pred, cost_fn=10, cost_fp=1):
    """
    Building cost matrix with customized cost
    """
    
    cost = 0
    for i, ii in zip(y, y_pred):
        if i == 1 and ii == 0:
            cost -= cost_fn
        elif i == 0 and ii == 1:
            cost -= cost_fp
    return cost

# Calculate total cost for one datapoint/event
def calc_cost(lst):
    init = 0
    for ticket in lst:
        init += ticket['quantity_sold'] * ticket['cost']
    return init

# Create cost matrix that corresponds to money saved
def build_cost_matrix_2(y, y_pred):
    init = 0
