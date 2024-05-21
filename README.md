Sure! Here is a markdown documentation for each step of building a decision tree model to predict whether a hotel booking is canceled or not.

# Predicting Hotel Booking Cancellations

## Problem Statement:
We need to build a decision tree model to predict whether a hotel booking is canceled or not using the provided hotel booking dataset.

## Data Description:
- **hotel:** Hotel type (H1 = Resort Hotel or H2 = City Hotel)
- **is_canceled:** Indicates if the booking was canceled (1) or not (0)
- **lead_time:** Days between the booking date and the arrival date
- **arrival_date_year:** Year of arrival
- **arrival_date_month:** Month of arrival
- **arrival_date_week_number:** Week number of the year for the arrival date
- **arrival_date_day_of_month:** Day of the month of arrival
- **stays_in_weekend_nights:** Number of weekend nights (Saturday or Sunday) stayed or booked
- **stays_in_week_nights:** Number of week nights (Monday to Friday) stayed or booked
- **adults:** Number of adults
- **children:** Number of children
- **babies:** Number of babies
- **meal:** Type of meal booked
  - Undefined/SC – No meal package
  - BB – Bed & Breakfast
  - HB – Half board (breakfast and one other meal, usually dinner)
  - FB – Full board (breakfast, lunch, and dinner)
- **country:** Country of origin (ISO 3155–3:2013 format)
- **market_segment:** Market segment designation
  - "TA" means "Travel Agents"
  - "TO" means "Tour Operators"
- **distribution_channel:** Booking distribution channel
  - "TA" means "Travel Agents"
  - "TO" means "Tour Operators"
- **is_repeated_guest:** Indicates if the booking name was from a repeated guest (1) or not (0)
- **previous_cancellations:** Number of previous bookings canceled by the customer
- **previous_bookings_not_canceled:** Number of previous bookings not canceled by the customer
- **reserved_room_type:** Code of reserved room type (anonymized)
- **assigned_room_type:** Code of assigned room type (may differ from reserved room type due to operational reasons or customer requests; anonymized)
- **booking_changes:** Number of changes/amendments made to the booking
- **deposit_type:** Type of deposit made
  - No Deposit – No deposit made
  - Non Refund – Deposit made equal to the total stay cost
  - Refundable – Deposit made less than the total stay cost
- **agent:** ID of the travel agency that made the booking (anonymized)
- **company:** ID of the company/entity responsible for the booking or payment (anonymized)
- **days_in_waiting_list:** Number of days the booking was on the waiting list
- **customer_type:** Type of booking
  - Contract – Booking has an allotment or contract associated
  - Group – Booking is associated with a group
  - Transient – Booking is not part of a group or contract
  - Transient-party – Transient booking associated with at least one other transient booking
- **adr:** Average Daily Rate (total lodging transactions divided by total staying nights)
- **required_car_parking_spaces:** Number of car parking spaces required
- **total_of_special_requests:** Number of special requests (e.g., twin bed, high floor)
- **reservation_status:** Last status of the reservation
  - Canceled – Booking was canceled
  - Check-Out – Customer checked in and departed
  - No-Show – Customer did not check in and did not inform the hotel
- **reservation_status_date:** Date when the last reservation status was set

## Step 1: Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

## Step 2: Load the Dataset

```python
df = pd.read_csv('/content/sample_data/hotel (6).csv')
df
```

## Step 3: Data Preprocessing

### 3.1 Define Encoding Dictionaries

```python
hotel_enc = {
    'Resort Hotel': 1,
    'City Hotel': 2
}
meal_enc = {
    'Undefined': 0,
    'SC': 1,
    'BB': 2,
    'HB': 3,
    'FB': 4
}
market_segment_enc = {
    'Direct': 1,
    'Corporate': 2,
    'Online TA': 3,
    'Offline TA/TO': 4,
    'Complementary': 5,
    'Groups': 6,
    'Undefined': 0
}
distribution_channel_enc = {
    'Direct': 1,
    'Corporate': 2,
    'TA/TO': 3,
    'Undefined': 0
}
deposit_type_enc = {
    'No Deposit': 1,
    'Non Refund': 2,
    'Refundable': 3
}
customer_type_enc = {
    'Contract': 1,
    'Group': 2,
    'Transient': 3,
    'Transient-party': 4
}
reservation_status_enc = {
    'Canceled': 1,
    'Check-Out': 2,
    'No-Show': 3
}
```

### 3.2 Apply the Encodings Using the Dictionaries

```python
df['hotel'] = df['hotel'].map(hotel_enc)
df['meal'] = df['meal'].map(meal_enc)
df['market_segment'] = df['market_segment'].map(market_segment_enc)
df['distribution_channel'] = df['distribution_channel'].map(distribution_channel_enc)
df['deposit_type'] = df['deposit_type'].map(deposit_type_enc)
df['customer_type'] = df['customer_type'].map(customer_type_enc)
df['reservation_status'] = df['reservation_status'].map(reservation_status_enc)
```

### 3.3 Handle Missing Values

```python
df.isna().sum()  # Check for missing values
df.fillna(0, inplace=True)  # Fill missing values with 0
```

### 3.4 Convert Date Columns to Datetime

```python
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
```

## Step 4: Exploratory Data Analysis (EDA)

### 4.1 Histograms and Boxplots

```python
# Histograms
df.hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplots
df.plot(kind='box', subplots=True, layout=(5, 6), figsize=(15, 10), patch_artist=True)
plt.tight_layout()
plt.show()
```

### 4.2 Value Counts

```python
value_counts_dict = {}
for col in df.columns:
    counts = df[col].value_counts().reset_index()
    counts.columns = ['Value', 'Count']
    value_counts_dict[col] = counts

for col, counts_df in value_counts_dict.items():
    print(f"Value Counts for Column: {col}")
    print(counts_df)
    print("\n")
```

### 4.3 Encoding the `arrival_date_month` Column

```python
month_enc = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}
df['arrival_date_month'] = df['arrival_date_month'].map(month_enc)
```

### 4.4 Compute the Correlation Matrix

```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
```

## Step 5: Handling Outliers

### 5.1 Detect Outliers Using IQR

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
print("Outliers per column:")
print(outliers.sum())  # Print the count of outliers per column
```

### 5.2 Optionally Remove Outliers

```python
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

## Step 6: Feature Transformation

### 6.1 Quantile Transform

```python
numerical_columns = ['lead_time', 'arrival_date_year', 'arrival_date_week_number',
                     'arrival_date_day_of_month', 'stays_in_weekend_nights',
                     'stays_in_week_nights', 'adults', 'children', 'babies',
                     'previous_cancellations', 'previous_bookings_not_canceled',
                     'booking_changes', 'days_in_waiting_list', 'adr',
                     'required_car_parking_spaces', 'total_of_special_requests']
```
# Apply quantile_transform to numerical columns

### 6.2 One-Hot Encoding for Categorical Variables

```python
# Define categorical columns for one-hot encoding
categorical_columns = ['reserved_room_type', 'assigned_room_type']

# Perform one-hot encoding
df = pd.get_dummies(df, columns=categorical_columns)
```

## Step 7: Splitting the Data

```python
# Define features and target variable
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## Step 8: Building the Decision Tree Model

```python
# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=0)

# Fit the model on the training data
dt_classifier.fit(X_train, y_train)
```

## Step 9: Making Predictions

```python
# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test)
```

## Step 10: Evaluating the Model

```python
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

This completes the process of building a decision tree model for predicting hotel booking cancellations.
