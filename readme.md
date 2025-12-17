# Sparse Manifold Learning via SOM

Topology-preserving mapping of high-dimensional sparse telecom data (DPI bitmaps → sparse vectors) using Self-Organizing Maps for non-convex quantization error minimization.


## SOM Optimization Dynamics

**Best Matching Unit (BMU):**  
c(t) = argmin_j || x(t) − w_j(t) ||_2

**Weight update:**  
w_j(t + 1) = w_j(t) + α(t) · h_{c,j}(r(t)) · ( x(t) − w_j(t) )

**Learning rate decay:**  
α(t) = α₀ / ( 1 + t / τ )

**Gaussian neighborhood function:**  
h_{c,j}(r) = exp( − r² / ( 2 · σ(t)² ) )

**Shrinking radius:**  
r(t) = r₀ · ( 1 − t / T )


**vs. Alternatives**:
| Method | Complexity | Topology Preservation | Scalability |
|--------|------------|----------------------|-------------|
| SOM    | \(O(n \cdot grid)\) | Non-linear manifolds | Batch-friendly |
| PCA    | \(O(d^2 n)\) | Linear only | Single pass |
| t-SNE  | \(O(n^2)\) | Non-linear | Small data only |

## Feature Sparsification Pipeline

Bitmap encoding sparsifies categorical lists (DPI policies, content types, IP protocols) → sparse binary vectors. Median imputation + z-score scaling for numerical features.

**Input**: Parquet with `SubscriberID`, `DpiPolicy[list]`, `contentType[list]`, `bytesFromClient/Server`, `sessions_count`, `transactionDuration`.

## File Structure

├── models2/

│ ├── som_model.joblib # SOM weights

│ ├── cluster_offers.joblib # Node → offer mapping

│ ├── policy_to_bit.joblib # DPI encoding

│ ├── content_to_bit.joblib # Content encoding

│ ├── proto_to_bit.joblib # Protocol encoding

│ ├── scaler.joblib # Standardization

│ ├── medians.joblib # Imputation

│ └── numeric_cols.joblib # Numeric features

├── notebook.ipynb # Derivations + full pipeline

├── main_sub1.py # Single inference

├── main_sub_multi.py # Batch processing

├── Dockerfile

└── results/ # U-matrix, quantization error decay


## Core Functions

- `load_models()`: Loads SOM + preprocessing pipeline
- `recommend_nbo()`: SOM distance → cluster → offer mapping
- `preprocess_customer_data()`: Bitmap sparsification + scaling
- `process_all_customers()`: Batch manifold projection

## Usage

python main_sub_multi.py # Batch → SQLite/CSV


**Output**: Recommendations with fallbacks (F3000G100M, F1200G50M, F3000G200M)

## Performance

Batch processing with tqdm progress, periodic SQLite commits for \(10^6+\) customers.

## License

MIT
