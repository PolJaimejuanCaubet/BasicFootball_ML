import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Loading Dataset
# Create target object and call it y, ELEMENT THAT YOU WANT TO PREDICT
# Create X ELEMENTS THAT YOU WANT/THINK/HOPE THAT ARE NEED IT TO PREDICT THAT ELEMENT
# Split into validation and training data
# Specify Model - RandomForestRegressos, DecisionTreeRegressor
# Fit Model, with trianed x and y
# Make validation predictions
# Calculate mean absolute error
# Using best value for max_leaf_nodes

league_dataset_path = 'C:\\Users\\huawei\\Desktop\\Basic ML\\LeaguesDataset\\LaLiga.csv'
league_dataset = pd.read_csv(league_dataset_path)

# Null Values being dropped
league_dataset.dropna(inplace=True) 

# Column Date to numerical features
league_dataset['Date'] = pd.to_datetime(league_dataset['Date'], format='%d/%m/%Y', errors='coerce')
league_dataset.dropna(subset=['Date'], inplace=True) 
league_dataset['Day'] = league_dataset['Date'].dt.day
league_dataset['Month'] = league_dataset['Date'].dt.month
league_dataset['Year'] = league_dataset['Date'].dt.year
league_dataset.drop('Date', axis=1, inplace=True)

# Coding categorical columns 'HomeTeam' and 'AwayTeam' correctly
label_encoder_home = LabelEncoder()
label_encoder_away = LabelEncoder()
league_dataset['HomeTeam'] = label_encoder_home.fit_transform(league_dataset['HomeTeam'])
league_dataset['AwayTeam'] = label_encoder_away.fit_transform(league_dataset['AwayTeam'])


# Convert 'Result' column to numerical values
label_encoder_result = LabelEncoder()
league_dataset['Result'] = label_encoder_result.fit_transform(league_dataset['Result'])

# print(league_dataset.Result)

features = ['HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'Day', 'Month', 'Year']
X = league_dataset[features]
y = league_dataset['Result']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

football_model = DecisionTreeRegressor(random_state=1)
football_model.fit(train_X, train_y)

validation_predictions = football_model.predict(val_X)

val_mae = mean_absolute_error(validation_predictions, val_y)
print(f"Mean Absolute Error: {val_mae}")

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [2,3,4,5,8,10,11,12,13,14,15,16,17,18,20]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {}

for leaf_size in candidate_max_leaf_nodes:
    mae = get_mae(leaf_size, train_X, val_X, train_y, val_y) 
    scores[leaf_size] = mae 

print(scores)  # Muestra los resultados
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X,y)

final_validation_predictions = football_model.predict(val_X)

print(final_validation_predictions)

# Convert the numerical predictions back to categorical values
predicted_results = label_encoder_result.inverse_transform(final_validation_predictions.astype(int))

# Get the corresponding home and away teams from the validation set
home_teams = label_encoder_home.inverse_transform(val_X['HomeTeam'])
away_teams = label_encoder_away.inverse_transform(val_X['AwayTeam'])

# Print the teams and the predicted results
for i in range(len(predicted_results)):
    print(f"Match: {home_teams[i]} vs {away_teams[i]} - Predicted Result: {predicted_results[i]}")
