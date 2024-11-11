# Module 7 Notes: Metrics and Model Development

## Metrics

Metrics should be `unbiased`, universal and concise.

1. A way to obtain similar responses
2. A way to measure performance
3. A way to measure prediction.

For our sample analysis we will use `KNN` K-Nearest Neighbor
- `K` is an arbitrary chosen number
- Need a `base case`
- Compare the neighbors
- Sort the results

Dataset for this analysis: 
```bash
icarus.cs.weber.edu:~hvalle/cs4580/movies.csv
```

## KNN-Euclidean Distance

Note: All code will be in `knn_analysis.py`


The Euclidean distance is the distance between points in `N-dimensional` space.

Formula
$
d(p, q) = \sqrt{\sum_{i=1}^n (q_i - p_i)^2}
$
where:
- $p = (p_1, p_2, \dots, p_n)$
- $q = (q_1, q_2, \dots, q_n)$

#### Task: 
Find the distance between these points:
- x = (0,0)
- y = (4,4)

Distance = 5.65685...

```python
# see
def euclidean_distance()
```

### KNN with Jaccard Similarity Index
Compares members of two individual sets to determine which members are `shared` and which are `distinct`. The index measures the similarity between the two sets. 

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Ex: $A = {1, 2, 3, 4}$ and $B = {3, 4, 5, 6}$ = $\frac{2}{6}$ or $0.33$

```python
# see
def jaccard_similarity_normal()
```

### KNN with Weighted Jaccard Similarity Index
The traditional Jaccard works well when doing `one-to-one` comparisons between a category. 

One solution is the `weighted` version. 
- Build a dictionary for `each genre` of the movies in our preferred list 

```python
# see
def jaccard_similarity_weighted()
```


### KNN with Levenshtein Distance
This is the most common form of edit-based metric, which generally quantifies to work required to transform a string from an initial sequence to a target sequence
-   It is used to determine the difference between two sequences(strings)
-   It is the distance between two words (minimum number of digits edits)
    -   insertions, deletions, or substitutions
$$
D(i,j) = 
\begin{cases}
j & \text{if } i = 0 \\
i & \text{if } j = 0 \\
D(i-1, j-1) & \text{if } s[i] = t[j] \\
1 + \min \{D(i-1,j), D(i, j-1), D(i-1,j-1)\} & \text{if } s[i] \neq t[j]
\end{cases}
$$

#### For example:
Consider these strings:
- s = 'kitten'
- t = 'sitting'

Find the `Levenshteain` distance
1.  Substitute `k` with `s` in `kitten` -> `sitten` (1 substituion)
2.  Substitute `e` with `i` in `sitten` -> `sittin` (1 substituion)
3.  Insert `g` at the end of `sittin` -> `sitting` (1 insertion)
   
Result is 3 edits, so the distance is $ = 3$


```python
# see
def knn_levenshtein_title()
```

# Need this package

```bash
# VE must be running python 3.11 or less
pip install Levenshtein
conda install levenshtein
```

### KNN Cosine Similarity Distance
It is used to measure the cosine of the angle between two vectors in a multi-dimensional space. This is commonly used in text analysis to measure similarities between documents


$$
\text{Cosine Similarity} = \cos(\theta)  
= \frac{A \cdot B}{|A| |B|} 
= \frac{\sum_{i=1}^{n} A_i B_i} { \sqrt{sum_{i=1}^n} A_i^2 \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}
$$
where 
- $ A \cdot B$ is the dot product of vectors $A$ and $B$
- $|A|$ and $|B|$ are the magnitude (or Euclidean norms) of vectors $A$ and $B$


### KNN Combining Metrics and Filtering Conditions

Two main concerns with `filtering`:
- Make too complicated (hard SQL queries)
- Too strict (end up with no results)

Combine `metrics` to generate `one` result:
- weight each metric
  - Should metrics contribute 50:50 or not
- Normalization of the combined metric
  - Make sure they have the same range

For our example we will use:
- `Cosine`: Use 20% of the `plot`
- Weighted Jaccard : Use 80% of `genre`

```python
# See 
cosine_and_weighted_jaccard()
```

### Prediction Metrics
A `prediction` is as simple as a guess about what is going to transpire
One prediction is `yes` or `no`. Binary.

How do we measure `accuracy` of the prediction?

```python
accuracy_metric.py
```



### Confusion Matrix
It is performed to measure how well your classification model is.
The model could be `binary` or `multi-class`. Each entry in a confusion 
matrix represents specific combination of predicted vs actual results.

For binary classificaitono, you have `four` parts
- True Positive (TP): Correctly predicted positive observations
- True Negative (TN): Correctly predicted negative observations
- False Positive (FP): Incorrecly predicted positive obserations (`Type 1 Error`)
- False Negative (FN): Incorrectly predicted negative observations (`Type 2 Error`)

Strucutre of the Matrix

|      




Key Metrics
- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`


```python
# See
confusion_matrix.py
```


