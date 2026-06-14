# Wealth Management Portfolio Risk Monitor - Walkthrough

We have successfully developed and verified the **AURA RiskMonitor** Web Dashboard. The application is a self-contained, real-time risk diagnostic utility that models daily asset return behavior, decomposes correlation structures using **Principal Component Analysis (PCA)**, forms exposure clusters using **DBSCAN**, and detects outlier risk isolation using **K-Nearest Neighbors (KNN)**.

---

## 1. Developed Modules & Core Architecture

The dashboard is structured into three clean files in the workspace:
- [index.html](file:///e:/Wealth%20Management/index.html): Structured grid layouts, parameter sliders, simulator forms, table metrics, and slide-out detail drawers.
- [index.css](file:///e:/Wealth%20Management/index.css): Sleek, dark-mode CSS design system using glassmorphism borders (`rgba(255,255,255,0.07)`), typography, transitions, customized sliders, scrollbars, and status badges.
- [app.js](file:///e:/Wealth%20Management/app.js): Core analytics engine handling return generation, PCA, DBSCAN, KNN, dynamic chart bindings, and user simulation actions.

### Underlying Algorithms Implementation
1. **Portfolio Return Simulation**: Models 30 major assets across 5 sectors over a 100-day historical window. Returns are generated using a multi-factor beta structure plus custom-injected anomalies (e.g. style decoupling for CRM, volatility drift for WFC, and phase anti-correlation for OXY).
2. **PCA Decomposition (Jacobi Algorithm)**: Computes the symmetric $30 \times 30$ covariance matrix of mean-centered daily returns. Solves for eigenvalues and eigenvectors exactly using the **Jacobi rotation sweep method** in client-side Javascript. It computes:
   - Cumulative explained variance ratio.
   - PCA projections ($X_c \cdot V_M$) and reconstructions ($Y \cdot V_M^T$).
   - Asset residuals ($E = X_c - X_r$) and corresponding L2-norm reconstruction error percentages representing idiosyncratic risk.
3. **DBSCAN Clustering**: Standardizes asset PC1 factor loadings, PC2 factor loadings, and daily returns volatilities. Group assets dynamically based on distance threshold Epsilon ($\epsilon$) and density threshold MinSamples. Noise points (-1) signify density-based anomalies.
4. **KNN Neighborhood Scoring ($K=5$)**: Measures individual asset isolation by calculating the average Euclidean distance to the 5 nearest risk exposure neighbors. Higher neighborhood distances signify un-hedged risk exposure.

---

## 2. Browser Verification & Test Results

The dashboard has been fully verified using automated browser subagents. The tests confirm:
- **Zero console errors**: Script loads and compiles perfectly, executing high-dimensional matrix rotations in sub-milliseconds.
- **Dynamic re-computations**: Modifying sliders (e.g. changing PCA systematic factor components or adjusting the KNN threshold) instantly triggers updates across all canvas charts and tables.
- **Click interactivity**: Clicking asset rows inside the **Risk Classification & Exposure Log** slides open the detail drawer, populating the individual returns chart, risk gauges, and custom risk-mitigation advisory texts.
- **Stress Simulator propagation**: Successfully verified that injecting an idiosyncratic shock on AAPL updates its risk status from "Safe" to "Critical", triggers L2 residual spikes, and reclassifies it as a DBSCAN noise outlier.

---

## 3. Visual Gallery

### Initial Dashboard Load
When the dashboard is first loaded, the pre-injected anomalies (`CRM`, `WFC`, `OXY`) are successfully detected:
![Dashboard Initial Load](/C:/Users/Friends/.gemini/antigravity-ide/brain/7e9b9f1b-e48e-4c41-b228-d1ddc6286501/dashboard_loaded_1781260226769.png)

### Asset Selection Detail Drawer
Clicking on `CRM` opens the sliding detail drawer on the right. The drawer renders the historical daily returns line plot, calculated statistics, and a custom advisory recommending scale-down due to high idiosyncratic decoupling:
![CRM Analysis Drawer](/C:/Users/Friends/.gemini/antigravity-ide/brain/7e9b9f1b-e48e-4c41-b228-d1ddc6286501/crm_details_drawer_1781260315469.png)

### Risk Stress Simulator (Shock Injection)
Injecting a $+7\sigma$ idiosyncratic shock to `AAPL` updates its daily return feed, causing its PCA residual to spike, its KNN isolation score to jump, and its DBSCAN label to become an outlier. It immediately changes status in the log table:
![AAPL Shocked State](/C:/Users/Friends/.gemini/antigravity-ide/brain/7e9b9f1b-e48e-4c41-b228-d1ddc6286501/aapl_shocked_1781260399666.png)

---

## 4. Test Recordings

The full step-by-step interactive sessions recorded during browser verification are saved as WebP animations in the artifact directory:
- **UI Exploration & Detail Drawer Test**: ![Dashboard Navigation Recording](/C:/Users/Friends/.gemini/antigravity-ide/brain/7e9b9f1b-e48e-4c41-b228-d1ddc6286501/dashboard_test_1781260200258.webp)
- **Stress Simulator Shock Test**: ![Simulator Shock Test Recording](/C:/Users/Friends/.gemini/antigravity-ide/brain/7e9b9f1b-e48e-4c41-b228-d1ddc6286501/shock_test_1781260350301.webp)
