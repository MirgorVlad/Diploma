from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_set(df: pd.DataFrame, target) -> Tuple:
	# Feature selection
	X = df.drop(columns=[target])

	# Target variable
	y = df[target]

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test

