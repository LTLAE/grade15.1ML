import pandas as pd

housing_dataset = pd.read_csv('exp1_housing.csv')

# Print all value types in the column 'ocean_proximity'
ocean_proximity_unique_values = housing_dataset['ocean_proximity'].unique()
print(ocean_proximity_unique_values)
# ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND']

# Add new columns for each unique value in 'ocean_proximity'
housing_dataset['NEAR BAY'] = 0
housing_dataset['<1H OCEAN'] = 0
housing_dataset['INLAND'] = 0
housing_dataset['NEAR OCEAN'] = 0
housing_dataset['ISLAND'] = 0

# Fill these columns according to column ocean_proximity
housing_dataset.loc[housing_dataset['ocean_proximity'] == 'NEAR BAY', 'NEAR BAY'] = 1
housing_dataset.loc[housing_dataset['ocean_proximity'] == '<1H OCEAN', '<1H OCEAN'] = 1
housing_dataset.loc[housing_dataset['ocean_proximity'] == 'INLAND', 'INLAND'] = 1
housing_dataset.loc[housing_dataset['ocean_proximity'] == 'NEAR OCEAN', 'NEAR OCEAN'] = 1
housing_dataset.loc[housing_dataset['ocean_proximity'] == 'ISLAND', 'ISLAND'] = 1

# Drop the original column 'ocean_proximity'
housing_dataset.drop('ocean_proximity', axis=1, inplace=True)

# Save the reformated dataset to a new file
housing_dataset.to_csv('exp1_housing_one_hot.csv', index=False)
