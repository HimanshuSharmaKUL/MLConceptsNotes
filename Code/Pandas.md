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
