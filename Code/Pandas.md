---
sticker: emoji//1f43c
---
https://skphd.medium.com/pandas-interview-questions-and-answers-a9e823a222c7
https://github.com/Devinterview-io/pandas-interview-questions

#### 1. How to filter  in Pandas
Conditional and boolean filtering
```python
#Using boolean indexing:
df[df['col_name']>value]

#Combing multiple conditions:
df[df['col1']>value1 & df['col2']<value2] #use bitwise operator in pandas: &,|
```
String filtering
```python
df[df['col'].str.contains('specific_string')]

#Handle case sensitivity:
df[df['col'].str.contains('specific_search_string', case=False)]
```

#### 1A. Sorting
```python
# Filter and sort the data
filtered_df = df[df['Department'].isin(['HR', 'IT'])]
sorted_df = df.sort_values(by='Age', ascending=False)
```

#### 2. Describe how _joining_ and _merging_ data works in _Pandas_.

##### Join: 
The `join` method is a convenient way to link DataFrames based on specific columns or their indices, aligning them either through intersection or union. Types of Joins
- **Inner Join**: Retains only the common entries between the DataFrames.
- **Outer Join**: Maintains all entries, merging based on where keys exist in either DataFrame.
```python
# Inner join on default indices
result_inner_index = df1.join(df2, lsuffix='_left')

# Inner join on specified column and index
result_inner_col_index = df1.join(df2, on='key', how='inner', lsuffix='_left', rsuffix='_right')
```

##### Merge: 
Greater flexibility than join, accommodating a range of keys to combine DataFrames
Types:
1. Left Merge: All entries from left are kept - matched with right df. Unmatched right df entries become NaN
2. Right Merge: similarly right
3. Outer Merge: Union of all entries. Mismatched values become NaN
4. Inner Merge: Selects only entries with matching keys in both left and right df
```python
# Perform a left merge aligning on 'key' column
left_merge = df1.merge(df2, on='key', how='left')

# Perform an outer merge on 'key' and 'another_key' columns
outer_merge = df1.merge(df2, left_on='key', right_on='another_key', how='outer')
```

#### 3. Rename, Modify column name in Pandas

```python
#Temporarily Rename
df.rename(columns = {'old_name': 'new_name'})
#Permanently Rename (in place)
df.rename(columns = {'old_name': 'new_name'}, inplace = True)
#Rename all columns
df.columns = ['new_name1', 'new_name2','new_name3']
```

#### 4. Handle duplicates
##### Identify Duplicate Rows: `duplicated()`
```python
import pandas as pd
# Create a sample DataFrame
data = {'A': [1, 1, 2, 2, 3, 3], 
		'B': ['a', 'a', 'b', 'b', 'c', 'c']}
df = pd.DataFrame(data)
# Identify duplicate rows
print(df.duplicated())  
# Output: [False, True, False, True, False, True]
```
##### Counting Duplicates
**count the occurrences of duplicates**, use the `duplicated()` method in conjunction with `sum()`. This provides the number of duplicates for each row.
```python
# Count duplicates
num_duplicates = df.duplicated().sum()
```
##### Drop Duplicates: `drup_duplicates()`
```python
# Drop duplicates
unique_df = df.drop_duplicates()
# Alternatively, you can keep the last occurrence
last_occurrence_df = df.drop_duplicates(keep='last')
# To drop in place, use the `inplace` parameter
df.drop_duplicates(inplace=True)
```
##### Aggregating
For numerical data, you can **aggregate** using functions such as mean or sum. This is beneficial when duplicates may have varying values in other columns.
```python
# Aggregate using mean
mean_df = df.groupby('A').mean()
```
##### Keep first/last Occurrence
```python
#By default, `drop_duplicates()` keeps the **first occurrence** of a duplicated row.
# Keep the last occurrence
df_last = df.drop_duplicates(keep='last')
```
##### Dropping Duplicate only from a subset of columns
Identifying duplicates might require considering a subset of columns. For instance, in an orders dataset, two orders with the same order date might still be distinct because they involve different products. **Set** the subset of columns to consider with the `subset` parameter.
```python
# Consider only the 'A' column to identify duplicates
df_unique_A = df.drop_duplicates(subset=['A'])
```

#### 5. Aggregating
- **`groupby()`**: uses Split-Apply-Combine technique i.e. `.groupby("column_name")` splits a dataframe into groups based on specified keys, often column names, then apply a function to each group and combine the results.
- aggregation functions: common ones -> `.sum()`, `.mean()`, `.median()`, `.count()`, and `.std()`
- Chaining GroupBy Methods: `.filter()`, `.apply()`, `.transform()`, `.assign()`.
- 
Example: Statistical Consulting
```python
sales_RFM = sales.groupby('CustomerId').agg({
    'ProductId': lambda y: y.mode(),
    'NumberOfRides': lambda y: y.sum(), #TotalRides
    'PurchaseDate': lambda y: (sales['PurchaseDate'].max()-y.max()).days, #Recency
    #'Price' : lambda y: len(y), #Kitni bari khareeda : Frequency
    'Price' : lambda y: round(y.sum(),2),  #Total kitne kharch kare: Monetary
    #'Price' : lambda y: round(y.mean(),2), #Average kitna khareeda
    #'Price' : lambda y: round(y.mode(), 2) #Sabse jyada kya khareeda
})

def f(x, sales):
    y={}
    y['ProductId_1'] = x['ProductId'].value_counts().index[0] #Most brought product
    y['TotalRides'] = x['NumberOfRides'].sum() #Total Rides
    y['Recency'] = (sales['PurchaseDate'].max()-x['PurchaseDate'].max()).days
    y['Frequency'] = len(x['Price'])
    y['Monetary'] = round(x['Price'].sum(), 2)
    y['AvgSpend'] = round(x['Price'].mean(), 2)
    return pd.Series(y, index=['ProductId_1', 'TotalRides', 'Recency', 'Frequency', 'Monetary', 'AvgSpend'])

sales_RFM = sales.groupby('CustomerId').apply(f, sales=sales)
```

Example: You call `.groupby()` and pass the name of the column that you want to group on, which is `["state", "gender"]`. Then, you use `["last_name"]` to specify the columns on which you want to perform the actual aggregation.
```python
n_by_state = df.groupby(["state", "gender"])["last_name"].count()
```

Example2: [1741. Find Total Time Spent by Each Employee](https://leetcode.com/problems/find-total-time-spent-by-each-employee/)
```python
def total_time(employees: pd.DataFrame) -> pd.DataFrame:
    return employees.assign(total_time = employees['out_time']-employees['in_time'],
                            day = employees['event_day'].dt.strftime('%Y-%m-%d'))
                    .groupby(["emp_id", "day"], as_index=False)[["total_time"]].sum()[["day","emp_id","total_time"]]
```

Example3;
```python

#If we do:
data[data.Year==2010].groupby(data['Week'], as_index=False)['Weekly_Sales'].mean()
#You're filtering the `data` for `Year == 2010`, then trying to group that **filtered DataFrame** by `data['Week']` — which comes from the **original unfiltered `data`**, not from the filtered subset. 
#- `data['Week']` might contain values that don’t exist in the filtered data
#- It's **not aligned** with `data[data.Year == 2010]` by index
#- So pandas can't guarantee the grouping is correct, and warns you about it

#So, Correct way to do it => use 'Week' instead of data['Week']
data[data.Year==2010].groupby('Week', as_index=False)['Weekly_Sales'].mean()
```

#### 6. Date Time
```python
data["Date"]=pd.to_datetime(data.Date) #OR
data["Date"]=pd.to_datetime(data["Date"])

data["Day"]=data.Date.dt.day
data["Month"]=data.Date.dt.month
data["Year"]=data.Date.dt.year
data['Week'] = data.Date.dt.isocalendar().week

# Changing the Months value from numbers to real values like Jan, Feb to Dec
import calendar
data['Month'] = data['Month'].apply(lambda x: calendar.month_abbr[x])
```
