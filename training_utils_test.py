__author__ = "Anthony (Tony) Kelly"
__copyright__ = "Copyright 2020, Northrop Grumman Corporation Mission Systems"
__organization__ = "Sector Analytics and Insights"
__python_version__ = "3.6.7"
__credits__ = ["Roderick Son, Mandeep Mundy, Dennis Tseng"]
__license__ = "NA"
__version__ = "0.0.1"
__maintainer__ = "Anthony Kelly"
__email__ = "anthony.kelly@ngc.com"
__status__ = "Development"

import numpy as np, pandas as pd, pyodbc, sqlalchemy, time
from sqlalchemy.sql import text as sa_text
from sklearn.metrics import confusion_matrix
from collections import Counter

# Function to collect required F1, Precision, and Recall Metrics
def collect_metrics(actuals, preds):
	# Create a confusion matrix
	matr = confusion_matrix(actuals, preds, labels=[0, 1])
	# Retrieve TN, FP, FN, and TP from the matrix
	true_negative, false_positive, false_negative, true_positive = confusion_matrix(
	    actuals, preds).ravel()

	# Compute precision
	precision = true_positive / (true_positive + false_positive)
	# Compute recall
	recall = true_positive / (true_positive + false_negative)

	# Return results
	return true_positive, false_positive, true_negative, false_negative

# Function to collect GSC Metrics
def collect_gsc_metrics(actuals, preds):
    on_time_count = 0
    late_count = 0
    on_time_correct = 0
    late_correct = 0

    for act, pred in zip(actuals, preds):
        # If it was on time
        if act == 0:
            on_time_count += 1
        # If it was late
        else:
            late_count += 1

        # Mark if it was correct on time
        if act == pred and act == 0:
            on_time_correct += 1

        # Mark if it was correct late
        if act == pred and act == 1:
            late_correct += 1

        # Calculate accuracies
        try:
            on_time_accuracy = on_time_correct / on_time_count
        except:
            on_time_accuracy = 0
        try:
            late_accuracy = late_correct / late_count
        except:
            late_accuracy = 0

    # Return accuracies
    return on_time_accuracy, late_accuracy

# Print the distribution of a numpy array
def print_distro(y):
    print(Counter(y))

# Print out metrics
def print_metrics(model_name, tp, fp, tn, fn):
    # Calculate true positive rates
    tpr = round(tp/(tp+fn), 3)*100
    fpr = round(fp/(fp+tn), 3)*100
    fnr = round(fn/(tp+fn), 3)*100
    tnr = round(tn/(fp+tn), 3)*100
    
    # Format prettyprint string
    print_str = """
    {} Predicted:
    -----------------------------------
    Late as Late: {}%
    OT As OT: {}%
    OT As Late: {}%
    Late as On Time: {}%
    """.format(model_name, tpr, tnr, fpr, fnr)
    
    # Print the print string
    print(print_str)

# Function to dummy encode categorical ORAD data
def dummy_encode(x):
    
    # Make a copy of the data that is passed in
    df = x.copy()

    # Make a list for each column that has _CD, _TYP, _ID, _NO, _STAT, _IND in the column name
    # Note , this method is called list comprehension
    cd_cols = [x for x in df.columns.values if ('_CD') in x]
    typ_cols = [x for x in df.columns.values if ('_TYP') in x]
    id_cols = [x for x in df.columns.values if ('_ID') in x]
    no_cols = [x for x in df.columns.values if ('_NO') in x]
    stat_cols = [x for x in df.columns.values if ('_STAT') in x]
    ind_cols = [x for x in df.columns.values if ('_IND') in x]

    # Concatenate the lists together
    encode_cols = cd_cols + typ_cols + id_cols + no_cols + stat_cols + ind_cols

    # Print the old size of the dataframe
    print("Old Size: {}".format(df.shape))
    # Print the # of numerical vs categorical columns
    print("# Numerical: {} | # Categorical: {}".format(df.shape[1] - len(encode_cols), len(encode_cols)))
    sizes = {}

    # For each column to encode
    for col in encode_cols:
        col_name = str(col)
        # Skip TIME_NO
        if col not in ('UNIQUE_ID', 'LATE_FRAME', 'ON_TIME', 'TIME_NO'):
            old_cols = df.shape[1]
            print("Now testing: {}".format(col_name))
            # Use pandas get_dummies function
            temp = pd.get_dummies(df[col], prefix=col_name, prefix_sep='_')
            # Do some fancy pandas magic
            df.drop(col, axis=1, inplace=True)
            df = pd.concat([df, temp], axis=1, join='inner')
            print("New Size: {}".format(df.shape))
            sizes[col] = df.shape[1] - old_cols
        else:
            continue
    
    # This is just like the list comprehension, except this makes a dictionary, so its dict comprehension
    new = {k: v for k, v in sorted(sizes.items(), key=lambda item: item[1], reverse=True)}
    
    #idxs = [df.columns.get_loc(x) for x in encode_cols]

    # Return the jawns
    return new, df    

# Function to query my SQL Server environment and retrieve data
def retrieve_data(query, verbose):
    # Establish a connection using pyodbc
    # Create a sqlalchemy engine object to issue sql statements to or from the DB                    
    engine = sqlalchemy.create_engine('mssql+pyodbc://riportai:PO_risk_project_2021@eim-db-ag40/j39304_ai_pro?driver=ODBC Driver 13 for SQL Server', fast_executemany = True) 

    # If verbose flag is enabled, keep track of how long the query takes
    if verbose:
        start = time.time()

    # Bring the data back in chunks to avoid timeout issues
    # Use list comprehension to create the df
    results_df = pd.concat(list([x for x in pd.read_sql(sql=query, con = engine, chunksize = 10000)]))

    # If verbose flag is enabled, print out how long it took
    if verbose:
        end = time.time()
        print("Query Complete In: {}".format(end-start))        
    
    # Return the results
    return results_df

# Function to query my SQL Server environment and retrieve data
def issue_query(query, verbose):
    # Establish a connection using pyodbc
    # Create a sqlalchemy engine object to issue sql statements to or from the DB                            
    engine = sqlalchemy.create_engine('mssql+pyodbc://riportai:PO_risk_project_2021@eim-db-ag40/j39304_ai_pro?driver=ODBC Driver 13 for SQL Server', fast_executemany = True) 

    # If verbose flag is enabled, keep track of how long the query takes
    if verbose:
        start = time.time()

    try:
        # Execute the sql statement
        result = engine.execute(query)
        engine.execute(sa_text(query).execution_options(autocommit=True))
        print("Executed query successfully")
    except:
        print("Failed to execute query")

    # If verbose flag is enabled, print out how long it took
    if verbose:
        end = time.time()
        print("Query Complete In: {}".format(end-start))  

# Function to query my SQL Server environment and retrieve data
def write_data(df, table_name, verbose):
    # Establish a connection using pyodbc
    # Create a sqlalchemy engine object to issue sql statements to or from the DB                        
    engine = sqlalchemy.create_engine('mssql+pyodbc://riportai:PO_risk_project_2021@eim-db-ag40/j39304_ai_pro?driver=ODBC Driver 13 for SQL Server', fast_executemany = True) 

    # If verbose flag is enabled, keep track of how long the query takes
    if verbose:
        start = time.time()

    # Use pandas' functionality to write to the database
    df.to_sql(name=table_name, index=False, con=engine, if_exists='append', chunksize=10000)

    # If verbose flag is enabled, print out how long it took
    if verbose:
        end = time.time()
        print("Query Complete In: {}".format(end-start))             