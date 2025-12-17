# Sparse Manifold Learning via SOM

Topology-preserving mapping of high-dimensional sparse telecom data (DPI bitmaps → sparse vectors) using Self-Organizing Maps (SOM) for non-convex quantization error minimization.

## SOM Optimization Dynamics

**Best Matching Unit (BMU):**  
\[ c(t) = \arg\min_j \|\mathbf{x}(t) - \mathbf{w}_j(t)\|_2 \]

**Weight Update:**  
\[ \mathbf{w}_j(t+1) = \mathbf{w}_j(t) + \alpha(t) \, h_{c,j}(r(t)) \, (\mathbf{x}(t) - \mathbf{w}_j(t)) \]

**Learning rate decay:**  
\[ \alpha(t) = \frac{\alpha_0}{1 + t/\tau} \]

**Gaussian neighborhood function:**  
\[ h_{c,j}(r) = \exp\left(-\frac{r^2}{2\sigma(t)^2}\right) \]

**Shrinking radius:**  
\[ r(t) = r_0 \left(1 - \frac{t}{T}\right) \]

## vs. Alternatives

| Method | Complexity      | Topology Preservation | Scalability      |
|--------|-----------------|----------------------|------------------|
| SOM    | \(O(n \cdot grid)\) | Non-linear manifolds | Batch-friendly  |
| PCA    | \(O(d^2 \cdot n)\)  | Linear only         | Single pass     |
| t-SNE  | \(O(n^2)\)      | Non-linear          | Small data only |

## Feature Sparsification Pipeline

Bitmap encoding converts categorical lists (DPI policies, content types, IP protocols) → sparse binary vectors. Median imputation + z-score scaling for numerical features.

**Input format**: Parquet with `SubscriberID`, `DpiPolicy[list]`, `contentType[list]`, `bytesFromClient/Server`, `sessions_count`, `transactionDuration`.

## File Structure

models2/
├── som_model.joblib # SOM weights
├── cluster_offers.joblib # Node → offer mapping
├── policy_to_bit.joblib # DPI encoding
├── content_to_bit.joblib # Content encoding
├── proto_to_bit.joblib # Protocol encoding
├── scaler.joblib # Standardization
├── medians.joblib # Imputation
└── numeric_cols.joblib # Numeric features

notebook.ipynb # Derivations + full pipeline
main_sub1.py # Single inference
main_sub_multi.py # Batch processing
Dockerfile
results/ # U-matrix, quantization error decay

text

## Core Functions

- `load_models()`: Loads SOM + preprocessing pipeline
- `recommend_nbo()`: SOM distance → cluster → offer mapping
- `preprocess_customer_data()`: Bitmap sparsification + scaling
- `process_all_customers()`: Batch manifold projection

## Usage

python main_sub_multi.py # Batch → SQLite/CSV

text

**Fallback offers**: F3000G100M, F1200G50M, F3000G200M

## Performance

Batch processing with tqdm progress, periodic SQLite commits for 10⁶+ customers.

## License

MIT
