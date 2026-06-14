# Wealth Management Portfolio Risk Monitoring Dashboard

A premium, interactive web application to monitor portfolio risk and detect anomalous asset behavior. The dashboard provides quantitative risk analytics using client-side implementations of **Principal Component Analysis (PCA) Residual Analysis**, **DBSCAN Clustering**, and **K-Nearest Neighbors (KNN) Neighborhood Scoring** (specifically $K=5$).

## Goal Description
In wealth management, detecting assets that behave atypically or represent hidden systemic exposure is critical for hedging and client protection.
This dashboard will:
1. **Simulate a realistic portfolio** of 30 assets across 5 distinct sectors (Tech, Finance, Healthcare, Energy, Utilities).
2. **Implement PCA Residual Analysis**: Decompose returns into systematic factor exposures and calculate reconstruction residuals. High residuals indicate high idiosyncratic (asset-specific) risk.
3. **Implement DBSCAN**: Group assets based on factor exposures and volatility to isolate outliers (density-based anomalies).
4. **Implement KNN Neighborhood Scoring ($K=5$)**: Compute the mean distance to the 5 nearest neighbors in return-risk space to identify isolated, un-hedged assets.
5. **Provide a stunning, responsive dark-mode interface** with real-time parameter controls, correlation heatmaps, asset detail cards, and an interactive "Inject Shock" simulator.

---

## User Review Required

> [!IMPORTANT]
> **Client-Side Algorithm Performance & Zero Dependencies**
> To avoid dependency issues, Python version differences, and server hosting complications, the core mathematical operations (Matrix algebra, Jacobi Eigenvalue algorithm, PCA, DBSCAN, and KNN) are implemented in pure JavaScript. This enables sub-millisecond computations directly in the browser and interactive real-time updates as parameters are tuned.
>
> We will load **Chart.js** via CDN for smooth, high-fidelity canvas charts, and **Lucide Icons** for modern iconography.

---

## Open Questions

> [!NOTE]
> 1. **Default PCA components**: We propose using **3 principal components** by default to capture market and major sector movements, leaving remaining variance as residuals. Is this choice aligned with your expectations?
> 2. **Mock dataset composition**: The simulation will model 30 major equities with sector-specific correlations and introduce 3 specific anomalies:
>    - *Idiosyncratic Shock Asset*: Normal return but suddenly high variance/uncorrelated moves.
>    - *Extreme Return Outlier*: High volatility and drift (easily caught by DBSCAN noise).
>    - *Unhedged Sector Asset*: Style drift, making it isolated in exposure space (caught by high KNN distance).
>    Is there any specific asset class or sector you would like to add (e.g., crypto, commodities)?

---

## Proposed Changes

We will create a self-contained, clean web app in `e:\Wealth Management`.

### [Frontend Component]

#### [NEW] [index.html](file:///e:/Wealth%20Management/index.html)
- Main layout structuring:
  - Header: Portfolio Risk Health Index and quick stats.
  - Sidebar: Parameter configuration (PCA Components, DBSCAN Epsilon/MinPts, KNN K-parameter, Anomaly Thresholds) and "Market Shock Simulator".
  - Grid Layout:
    - **Panel 1**: PCA Residual Analysis (scatter plot of volatility vs. reconstruction error).
    - **Panel 2**: DBSCAN Exposure Clustering (2D projection of factor exposures, highlighting noise points).
    - **Panel 3**: KNN Neighborhood Scoring (bar chart of 5-NN anomaly scores).
    - **Panel 4**: Risk Correlation Matrix & Anomaly Log.
  - Selected Asset Drawer: Detailed view of asset returns, sector exposure, and mitigation recommendations.

#### [NEW] [index.css](file:///e:/Wealth%20Management/index.css)
- Custom CSS design system:
  - **Aesthetics**: Sleek dark mode using an obsidian/slate background, glassmorphism card panels (`backdrop-filter: blur`, thin semi-transparent borders), glowing accent colors (emerald for normal, amber for warnings, vivid rose/red for anomalies).
  - Modern typography: Plus Jakarta Sans / Inter font families.
  - Hover states, micro-transitions (cards scale and glow on hover), responsive grids, scrollbar styling.

#### [NEW] [app.js](file:///e:/Wealth%20Management/app.js)
- Core JavaScript engine:
  - **Data Engine**: Generates 100 days of return time-series for 30 assets.
  - **PCA Module**: 
    - Computes covariance matrix.
    - Implements the **Jacobi Eigenvalue Algorithm** to extract eigenvectors and eigenvalues.
    - Projects data onto the first $M$ principal components.
    - Reconstructs returns and computes L2-norm reconstruction error (residual) for each asset.
  - **DBSCAN Module**: Density-based clustering of standardized risk features [PCA Loadings, Volatility, Mean Return].
  - **KNN Module**: Calculates Euclidean distance matrix and computes the average distance to the 5-nearest neighbors.
  - **Chart Controller**: Manages Chart.js instances, updating charts seamlessly when sliders change.
  - **Simulation Engine**: Triggers custom sector or asset shocks, updating data feeds in real-time.

---

## Verification Plan

### Automated/Code Verification
- Run a static server (e.g., using python's `http.server`) to serve the web application.
- Launch the browser subagent to evaluate:
  - HTML structure and syntax.
  - JS error-free execution (verifying algorithms return mathematically sound values matching expectation, e.g. sum of eigenvalues equals total variance, DBSCAN outputs valid clusters, KNN correctly finds nearest neighbors).

### Manual Verification
- Verify responsiveness and layout across different viewport sizes.
- Test interactive sliders: adjust PCA components, DBSCAN epsilon, and KNN threshold, verifying that chart plots update dynamically and outliers are re-classified immediately.
- Use the "Inject Shock" controls to confirm that the affected asset immediately jumps in PCA residual and KNN scoring, triggering the alarm system.
