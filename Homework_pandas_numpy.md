```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv("anaconda_projects/train.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
def main_info(col):
    range = col.max() - col.min()
    print(col.describe())
    print(pd.DataFrame({
        "Median": col.median(),
        "Mode": col.mode(),
        "Variation": col.var(),
        "Range": range
    }, index=[0]))

for i in df.select_dtypes(include='number'):
    main_info(df[i])
```

    count    891.000000
    mean     446.000000
    std      257.353842
    min        1.000000
    25%      223.500000
    50%      446.000000
    75%      668.500000
    max      891.000000
    Name: PassengerId, dtype: float64
       Median  Mode  Variation  Range
    0   446.0     1    66231.0    890
    count    891.000000
    mean       0.383838
    std        0.486592
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        1.000000
    max        1.000000
    Name: Survived, dtype: float64
       Median  Mode  Variation  Range
    0     0.0     0   0.236772      1
    count    891.000000
    mean       2.308642
    std        0.836071
    min        1.000000
    25%        2.000000
    50%        3.000000
    75%        3.000000
    max        3.000000
    Name: Pclass, dtype: float64
       Median  Mode  Variation  Range
    0     3.0     3   0.699015      2
    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64
       Median  Mode   Variation  Range
    0    28.0  24.0  211.019125  79.58
    count    891.000000
    mean       0.523008
    std        1.102743
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        1.000000
    max        8.000000
    Name: SibSp, dtype: float64
       Median  Mode  Variation  Range
    0     0.0     0   1.216043      8
    count    891.000000
    mean       0.381594
    std        0.806057
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        0.000000
    max        6.000000
    Name: Parch, dtype: float64
       Median  Mode  Variation  Range
    0     0.0     0   0.649728      6
    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64
        Median  Mode    Variation     Range
    0  14.4542  8.05  2469.436846  512.3292
    


```python
df.select_dtypes(include="number").corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>1.000000</td>
      <td>-0.005007</td>
      <td>-0.035144</td>
      <td>0.036847</td>
      <td>-0.057527</td>
      <td>-0.001652</td>
      <td>0.012658</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.005007</td>
      <td>1.000000</td>
      <td>-0.338481</td>
      <td>-0.077221</td>
      <td>-0.035322</td>
      <td>0.081629</td>
      <td>0.257307</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.035144</td>
      <td>-0.338481</td>
      <td>1.000000</td>
      <td>-0.369226</td>
      <td>0.083081</td>
      <td>0.018443</td>
      <td>-0.549500</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.036847</td>
      <td>-0.077221</td>
      <td>-0.369226</td>
      <td>1.000000</td>
      <td>-0.308247</td>
      <td>-0.189119</td>
      <td>0.096067</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.057527</td>
      <td>-0.035322</td>
      <td>0.083081</td>
      <td>-0.308247</td>
      <td>1.000000</td>
      <td>0.414838</td>
      <td>0.159651</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.001652</td>
      <td>0.081629</td>
      <td>0.018443</td>
      <td>-0.189119</td>
      <td>0.414838</td>
      <td>1.000000</td>
      <td>0.216225</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.012658</td>
      <td>0.257307</td>
      <td>-0.549500</td>
      <td>0.096067</td>
      <td>0.159651</td>
      <td>0.216225</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''Conclutions: We put target = df["Survived"], and can see that the highest corr
is with the column called Pclass. According to literature, first class had the
highest % of survivors: first of all, in rescue operation the advanteg had been
giving to them, second of all, they accepted a warning earlier then for example
people from the third class, and third of all, they could get the access to rescue
boats not only easyer, but also faster comparing to other classes.
Second highest corr is with the Fare, which is price for the ticket. Then higher the
price, then your class closer to 1, reffers to first selected corr'''
```


```python
# Numpy homework
```


```python
# Normal Distribution
import numpy as np
def normal_distribution(x, mu, sigma):
  first_part = np.divide(1, np.multiply(sigma, np.sqrt(2 * np.pi)))
  second_part = np.exp(np.multiply(-0.5, np.power(np.divide(np.subtract(x, mu), sigma), 2)))
  return np.multiply(first_part, second_part)
```


```python
# Sigmoid Function
def sigmoid(x):
  denominetor = np.add(1, np.exp(-x))
  return np.divide(1, denominetor)
```


```python
# Weights update in Logistic Regression
def w_update(w, X, y, y_h, alpha= 0.0005):
  n = X.shape[0]
  error = np.substract(y_h, y)
  gradient = np.dot(X.T, error)/n
  w = w - np.multiply(alpha, gradient)
  return w
```


```python
# MSE
def mse_f(y, y_h):
  error = np.substract(y_h, y)
  mse = np.mean(np.power(error, 2))
  return mse
```


```python
# Binary Cross Entropy
def bce(y, y_h):
  eps = 1e-10
  y_h = np.clip(y_h, eps, 1-eps)
  loss = -np.mean(y*np.log(y_h)+ np.multiply((1-y), np.log(1-y_h)))
  return loss
```


```python

```
