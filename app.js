/**
 * Wealth Management Portfolio Risk Monitor - Analytical Engine
 * AURA RiskMonitor (PCA, DBSCAN & KNN Anomaly Detection)
 */

// Global App State
const state = {
  assets: [], // Holds information, returns, and current analytical metrics for the 30 assets
  selectedAssetIndex: 0,
  pcaComponents: 3,
  dbscanEps: 1.8,
  dbscanMinSamples: 3,
  knnK: 5,
  knnThreshold: 2.2,
  pcaThreshold: 0.65, // Dynamic L2 residual cutoff
  charts: {
    pcaResidual: null,
    dbscanCluster: null,
    knnScore: null,
    assetDetail: null
  },
  randomSeed: 125 // Seed for deterministic mock returns
};

// Tick Tock: Asset definition
const ASSETS_CONFIG = [
  // Tech Sector (0-5)
  { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Technology', desc: 'Consumer Electronics & Cloud Services' },
  { ticker: 'MSFT', name: 'Microsoft Corp.', sector: 'Technology', desc: 'Enterprise Software & Cloud Platforms' },
  { ticker: 'GOOG', name: 'Alphabet Inc.', sector: 'Technology', desc: 'Digital Advertising & AI Services' },
  { ticker: 'NVDA', name: 'NVIDIA Corp.', sector: 'Technology', desc: 'Semiconductors & GPU Computing' },
  { ticker: 'AMD', name: 'Advanced Micro Devices', sector: 'Technology', desc: 'Microprocessors & Graphics Chips' },
  { ticker: 'CRM', name: 'Salesforce Inc.', sector: 'Technology', desc: 'Enterprise Cloud & CRM Platforms (Anomaly candidate)' },
  
  // Finance Sector (6-11)
  { ticker: 'JPM', name: 'JPMorgan Chase & Co.', sector: 'Finance', desc: 'Global Investment & Commercial Banking' },
  { ticker: 'BAC', name: 'Bank of America Corp.', sector: 'Finance', desc: 'Retail Banking & Financial Advisory' },
  { ticker: 'MS', name: 'Morgan Stanley', sector: 'Finance', desc: 'Wealth Management & Investment Banking' },
  { ticker: 'GS', name: 'Goldman Sachs Group', sector: 'Finance', desc: 'Securities Trading & Asset Management' },
  { ticker: 'C', name: 'Citigroup Inc.', sector: 'Finance', desc: 'Global Transaction & Retail Banking' },
  { ticker: 'WFC', name: 'Wells Fargo & Co.', sector: 'Finance', desc: 'Diversified Financial Services (Anomaly candidate)' },

  // Healthcare Sector (12-17)
  { ticker: 'JNJ', name: 'Johnson & Johnson', sector: 'Healthcare', desc: 'Medical Devices & Consumer Healthcare' },
  { ticker: 'PFE', name: 'Pfizer Inc.', sector: 'Healthcare', desc: 'Biopharmaceutical Drug Development' },
  { ticker: 'LLY', name: 'Eli Lilly & Co.', sector: 'Healthcare', desc: 'Endocrine & Oncology Therapeutics' },
  { ticker: 'UNH', name: 'UnitedHealth Group', sector: 'Healthcare', desc: 'Managed Healthcare & Insurance' },
  { ticker: 'MRK', name: 'Merck & Co. Inc.', sector: 'Healthcare', desc: 'Infectious Disease & Immuno-oncology' },
  { ticker: 'ABT', name: 'Abbott Laboratories', sector: 'Healthcare', desc: 'Diagnostics, Devices & Nutritionals' },

  // Energy Sector (18-23)
  { ticker: 'XOM', name: 'Exxon Mobil Corp.', sector: 'Energy', desc: 'Upstream Oil Exploration & Refining' },
  { ticker: 'CVX', name: 'Chevron Corp.', sector: 'Energy', desc: 'Integrated Petroleum Operations' },
  { ticker: 'COP', name: 'ConocoPhillips', sector: 'Energy', desc: 'Crude Oil Production & Mining' },
  { ticker: 'EOG', name: 'EOG Resources Inc.', sector: 'Energy', desc: 'Shale Oil & Natural Gas Exploration' },
  { ticker: 'SLB', name: 'Schlumberger Ltd.', sector: 'Energy', desc: 'Oilfield Technology & Services' },
  { ticker: 'OXY', name: 'Occidental Petroleum', sector: 'Energy', desc: 'Oil & Chemicals Exploration (Anomaly candidate)' },

  // Utilities Sector (24-29)
  { ticker: 'NEE', name: 'NextEra Energy Inc.', sector: 'Utilities', desc: 'Renewable wind & solar utilities' },
  { ticker: 'DUK', name: 'Duke Energy Corp.', sector: 'Utilities', desc: 'Regulated Electric & Gas Utility' },
  { ticker: 'SO', name: 'Southern Company', sector: 'Utilities', desc: 'Electric Power generation & transmission' },
  { ticker: 'AEP', name: 'American Electric Power', sector: 'Utilities', desc: 'Investor-owned electricity distributor' },
  { ticker: 'EXC', name: 'Exelon Corp.', sector: 'Utilities', desc: 'Nuclear Power Generation & Delivery' },
  { ticker: 'SRE', name: 'Sempra Energy', sector: 'Utilities', desc: 'Natural Gas and Electricity infrastructure' }
];

// Seeded LCG Random Number Generator
function lcgRandom(seedObj) {
  // LCG parameters
  const a = 1664525;
  const c = 1013904223;
  const m = Math.pow(2, 32);
  seedObj.seed = (a * seedObj.seed + c) % m;
  return seedObj.seed / m;
}

function randomNormal(seedObj) {
  // Box-Muller transform for normal distribution
  let u1 = 0, u2 = 0;
  while(u1 === 0) u1 = lcgRandom(seedObj); 
  while(u2 === 0) u2 = lcgRandom(seedObj);
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

/**
 * Step 1: Simulated Return Generator
 * Generates 100 trading days of returns for 30 assets.
 * Uses a Multi-Factor model: Returns = Beta_m * Market + Beta_s * Sector + Idiosyncratic_Noise
 */
function generatePortfolioReturns(customSeed = 125) {
  const seedObj = { seed: customSeed };
  const T = 100; // Time steps
  const N = 30; // Number of assets
  
  // Initialize returns array
  const assetReturns = Array(N).fill(0).map(() => Array(T).fill(0));
  
  // 1. Generate market factor and sector factors for 100 days
  const marketFactor = Array(T).fill(0).map(() => randomNormal(seedObj) * 0.008 + 0.0002);
  const sectorFactors = {
    'Technology': Array(T).fill(0).map(() => randomNormal(seedObj) * 0.006),
    'Finance': Array(T).fill(0).map(() => randomNormal(seedObj) * 0.005),
    'Healthcare': Array(T).fill(0).map(() => randomNormal(seedObj) * 0.004),
    'Energy': Array(T).fill(0).map(() => randomNormal(seedObj) * 0.009 - 0.0001),
    'Utilities': Array(T).fill(0).map(() => randomNormal(seedObj) * 0.003 + 0.0001)
  };

  // 2. Generate asset returns
  for (let i = 0; i < N; i++) {
    const asset = ASSETS_CONFIG[i];
    
    // Assign systematic beta weights based on sectors
    let betaM = 1.0; // Market beta
    let betaS = 0.8; // Sector beta
    
    if (asset.sector === 'Technology') { betaM = 1.3; betaS = 0.9; }
    else if (asset.sector === 'Finance') { betaM = 1.1; betaS = 0.7; }
    else if (asset.sector === 'Healthcare') { betaM = 0.8; betaS = 0.5; }
    else if (asset.sector === 'Energy') { betaM = 1.2; betaS = 1.1; }
    else if (asset.sector === 'Utilities') { betaM = 0.5; betaS = 0.4; }
    
    // Add minor asset-specific loading differences to prevent duplicate assets
    betaM += (lcgRandom(seedObj) - 0.5) * 0.2;
    betaS += (lcgRandom(seedObj) - 0.5) * 0.2;

    for (let t = 0; t < T; t++) {
      // Base idiosyncratic noise
      let idiosyncraticNoise = randomNormal(seedObj) * 0.005;
      
      // Inject Anomaly 1: CRM (Asset 5 - Tech)
      // Decoupled idiosyncratic risk - high variance, no market tracking
      if (i === 5) {
        betaM = 0.2; // Decouples from market
        betaS = 0.1; // Decouples from tech sector
        idiosyncraticNoise = randomNormal(seedObj) * 0.022; // 4.5x normal idiosyncratic variance
        // Add occasional shock spikes
        if (t === 25 || t === 60 || t === 85) {
          idiosyncraticNoise += 0.06;
        }
      }
      
      // Inject Anomaly 2: WFC (Asset 11 - Finance)
      // Drift & Volatility Outlier (Isolated point in risk space)
      if (i === 11) {
        betaM = 1.8; // High systematic exposure
        betaS = -0.5; // Opposite sector behavior
        idiosyncraticNoise = randomNormal(seedObj) * 0.025 - 0.005; // Large negative drift and volatility
      }

      // Inject Anomaly 3: OXY (Asset 23 - Energy)
      // Phase decoupled sector asset (High KNN distance, normal volatility)
      if (i === 23) {
        betaM = -0.6; // Anti-correlated with market
        betaS = -0.9; // Anti-correlated with Energy sector
        idiosyncraticNoise = randomNormal(seedObj) * 0.008;
      }
      
      assetReturns[i][t] = betaM * marketFactor[t] + betaS * sectorFactors[asset.sector][t] + idiosyncraticNoise;
    }
  }
  
  return assetReturns;
}

/**
 * Step 2: PCA Mathematical Core
 * Computes exact covariance matrix and solves for eigenvectors using Jacobi Eigenvalue Decomposition.
 */
function runPCA(returnsMatrix, numComponents = 3) {
  const N = returnsMatrix.length;
  const T = returnsMatrix[0].length;
  
  // 1. Mean-center returns for each asset
  const means = Array(N).fill(0);
  const centeredReturns = Array(N).fill(0).map(() => Array(T).fill(0));
  
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (let t = 0; t < T; t++) {
      sum += returnsMatrix[i][t];
    }
    means[i] = sum / T;
    
    for (let t = 0; t < T; t++) {
      centeredReturns[i][t] = returnsMatrix[i][t] - means[i];
    }
  }
  
  // 2. Compute Covariance Matrix (Size N x N)
  const covMatrix = Array(N).fill(0).map(() => Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let sum = 0;
      for (let t = 0; t < T; t++) {
        sum += centeredReturns[i][t] * centeredReturns[j][t];
      }
      const val = sum / (T - 1);
      covMatrix[i][j] = val;
      covMatrix[j][i] = val; // Symmetric
    }
  }
  
  // 3. Jacobi Eigenvalue Decomposition
  const jacobiResult = jacobiEigenvalues(covMatrix);
  const eigenvalues = jacobiResult.eigenvalues;
  const eigenvectors = jacobiResult.eigenvectors; // N x N matrix (columns are eigenvectors)
  
  // 4. Sort Eigenvalues and corresponding Eigenvectors descending
  const eigenPairs = eigenvalues.map((val, idx) => {
    return {
      eigenvalue: val,
      vector: eigenvectors.map(row => row[idx]) // Extract column eigenvector
    };
  });
  
  eigenPairs.sort((a, b) => b.eigenvalue - a.eigenvalue);
  
  const sortedEigenvalues = eigenPairs.map(p => p.eigenvalue);
  const sortedEigenvectors = eigenPairs.map(p => p.vector); // N eigenvectors, each size N
  
  // 5. Total variance & explained variance ratio
  const totalVariance = sortedEigenvalues.reduce((sum, v) => sum + v, 0);
  const explainedVariancePct = sortedEigenvalues.slice(0, numComponents).reduce((sum, v) => sum + v, 0) / totalVariance * 100;
  
  // 6. PCA Reconstruction & Residual Analysis
  // For each day, project centered returns onto first M principal components, then reconstruct.
  // Projection: Y = X_c * V_M (T x M)
  // Reconstruction: X_reconstruct = Y * V_M^T (T x N)
  // Residuals: E = X_c - X_reconstruct
  const reconstructionErrors = Array(N).fill(0);
  
  // Select first M eigenvectors (each size N)
  const components = sortedEigenvectors.slice(0, numComponents);
  
  for (let i = 0; i < N; i++) {
    let residualSumSq = 0;
    for (let t = 0; t < T; t++) {
      // Reconstructed value for asset i at day t:
      // Let scores Y[t][k] = sum_{j=0}^{N-1} centeredReturns[j][t] * component[k][j]
      // Reconstructed value = sum_{k=0}^{M-1} Y[t][k] * component[k][i]
      let reconstructedVal = 0;
      for (let k = 0; k < numComponents; k++) {
        const comp = components[k];
        let score_t_k = 0;
        for (let j = 0; j < N; j++) {
          score_t_k += centeredReturns[j][t] * comp[j];
        }
        reconstructedVal += score_t_k * comp[i];
      }
      
      const error = centeredReturns[i][t] - reconstructedVal;
      residualSumSq += error * error;
    }
    
    // L2 normalized reconstruction error (as percentage return)
    reconstructionErrors[i] = Math.sqrt(residualSumSq / T) * 100;
  }
  
  return {
    eigenvalues: sortedEigenvalues,
    eigenvectors: sortedEigenvectors,
    explainedVariancePct: explainedVariancePct.toFixed(1),
    residuals: reconstructionErrors,
    means: means
  };
}

/**
 * Jacobi Eigenvalue Algorithm for Symmetric Matrices
 * Finds all eigenvalues and eigenvectors.
 */
function jacobiEigenvalues(matrix, maxIter = 150) {
  const n = matrix.length;
  // A is symmetric, copy values
  let A = matrix.map(row => [...row]);
  
  // Initialize V as identity matrix (N x N)
  let V = Array(n).fill(0).map((_, i) => {
    let row = Array(n).fill(0);
    row[i] = 1;
    return row;
  });
  
  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element in absolute value
    let p = 0, q = 1;
    let maxVal = Math.abs(A[0][1]);
    
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[i][j]) > maxVal) {
          maxVal = Math.abs(A[i][j]);
          p = i;
          q = j;
        }
      }
    }
    
    // Convergence check
    if (maxVal < 1e-10) {
      break;
    }
    
    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    
    // Compute Jacobi rotation angle
    const tau = (aqq - app) / (2.0 * apq);
    let t;
    if (tau >= 0) {
      t = 1.0 / (tau + Math.sqrt(1.0 + tau * tau));
    } else {
      t = -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
    }
    
    const c = 1.0 / Math.sqrt(1.0 + t * t);
    const s = c * t;
    
    // Update diagonal elements of matrix A
    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0;
    A[q][p] = 0;
    
    // Update off-diagonal elements of matrix A
    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const aip = A[i][p];
        const aiq = A[i][q];
        
        A[i][p] = A[p][i] = c * aip - s * aiq;
        A[i][q] = A[q][i] = s * aip + c * aiq;
      }
    }
    
    // Update orthogonal eigenvector matrix V (rotate columns p and q)
    for (let i = 0; i < n; i++) {
      const vip = V[i][p];
      const viq = V[i][q];
      
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }
  
  // Extract eigenvalues from diagonal
  let eigenvalues = Array(n);
  for (let i = 0; i < n; i++) {
    eigenvalues[i] = A[i][i];
  }
  
  return { eigenvalues, eigenvectors: V };
}

/**
 * Step 3: Clustering & Outlier Detection (DBSCAN)
 * Standardizes risk features and performs Density-Based Clustering.
 */
function runDBSCAN(features, eps, minSamples) {
  const N = features.length;
  const labels = Array(N).fill(0); // 0 = unvisited, -1 = noise, 1+ = cluster ID
  let clusterId = 0;
  
  // Compute pairwise Euclidean distances
  const distMatrix = Array(N).fill(0).map(() => Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      if (i === j) {
        distMatrix[i][j] = 0;
      } else {
        const d = Math.sqrt(
          Math.pow(features[i][0] - features[j][0], 2) + 
          Math.pow(features[i][1] - features[j][1], 2) +
          Math.pow(features[i][2] - features[j][2], 2)
        );
        distMatrix[i][j] = d;
        distMatrix[j][i] = d;
      }
    }
  }
  
  // Core DBSCAN loop
  for (let i = 0; i < N; i++) {
    if (labels[i] !== 0) continue; // Skip already visited points
    
    // Find epsilon-neighbors of i
    const neighbors = [];
    for (let j = 0; j < N; j++) {
      if (distMatrix[i][j] <= eps) {
        neighbors.push(j);
      }
    }
    
    if (neighbors.length < minSamples) {
      labels[i] = -1; // Flag as Noise (Outlier) candidate
    } else {
      clusterId++;
      labels[i] = clusterId; // Start new cluster
      
      // Expand cluster
      const queue = [...neighbors];
      for (let k = 0; k < queue.length; k++) {
        const p = queue[k];
        
        // If labeled as Noise, convert to boundary point of this cluster
        if (labels[p] === -1) {
          labels[p] = clusterId;
        }
        
        if (labels[p] === 0) {
          labels[p] = clusterId; // Assign to cluster
          
          // Find neighbors of p
          const pNeighbors = [];
          for (let j = 0; j < N; j++) {
            if (distMatrix[p][j] <= eps) {
              pNeighbors.push(j);
            }
          }
          
          // If p is a core point, expand search queue
          if (pNeighbors.length >= minSamples) {
            for (let j = 0; j < pNeighbors.length; j++) {
              if (queue.indexOf(pNeighbors[j]) === -1) {
                queue.push(pNeighbors[j]);
              }
            }
          }
        }
      }
    }
  }
  
  return {
    labels: labels,
    distances: distMatrix
  };
}

/**
 * Step 4: KNN Neighborhood Scoring (K=5)
 * Calculates the mean distance to the K closest neighbors in standardized risk feature space.
 */
function runKNNNeighborhoodScoring(distMatrix, k = 5) {
  const N = distMatrix.length;
  const knnScores = Array(N).fill(0);
  
  for (let i = 0; i < N; i++) {
    // Collect distances to other assets (excluding self)
    const distances = [];
    for (let j = 0; j < N; j++) {
      if (i !== j) {
        distances.push(distMatrix[i][j]);
      }
    }
    
    // Sort distances ascending
    distances.sort((a, b) => a - b);
    
    // Take mean of K nearest neighbors
    let sum = 0;
    for (let j = 0; j < k; j++) {
      sum += distances[j];
    }
    knnScores[i] = sum / k;
  }
  
  return knnScores;
}

/**
 * Standardizes risk features: Mean 0, Variance 1
 */
function standardizeFeatures(rawFeatures) {
  const N = rawFeatures.length;
  const D = rawFeatures[0].length;
  
  const standardized = Array(N).fill(0).map(() => Array(D).fill(0));
  
  for (let d = 0; d < D; d++) {
    // Mean
    let sum = 0;
    for (let i = 0; i < N; i++) {
      sum += rawFeatures[i][d];
    }
    const mean = sum / N;
    
    // Standard deviation
    let sumSqDiff = 0;
    for (let i = 0; i < N; i++) {
      sumSqDiff += Math.pow(rawFeatures[i][d] - mean, 2);
    }
    const sd = Math.sqrt(sumSqDiff / N) || 1.0;
    
    // Standardize
    for (let i = 0; i < N; i++) {
      standardized[i][d] = (rawFeatures[i][d] - mean) / sd;
    }
  }
  
  return standardized;
}

/**
 * Main Data Pipeline
 * Runs PCA, DBSCAN, and KNN analysis on the portfolio returns.
 */
function processPortfolioData() {
  const returns = state.returns;
  const N = returns.length;
  const T = returns[0].length;
  
  // Calculate raw stats for each asset
  const volatilities = Array(N).fill(0);
  const avgReturns = Array(N).fill(0);
  
  for (let i = 0; i < N; i++) {
    // Mean
    let sum = 0;
    for (let t = 0; t < T; t++) {
      sum += returns[i][t];
    }
    avgReturns[i] = sum / T;
    
    // Daily standard deviation
    let sumSq = 0;
    for (let t = 0; t < T; t++) {
      sumSq += Math.pow(returns[i][t] - avgReturns[i], 2);
    }
    volatilities[i] = Math.sqrt(sumSq / (T - 1));
  }
  
  // 1. Run PCA
  const pcaResults = runPCA(returns, state.pcaComponents);
  
  // Construct raw risk features: [PC1 Loading, PC2 Loading, Volatility]
  // PC1 and PC2 loadings are column elements of the respective eigenvectors.
  const rawFeatures = Array(N).fill(0).map((_, i) => {
    return [
      pcaResults.eigenvectors[0][i], // PC1 Loading
      pcaResults.eigenvectors[1][i], // PC2 Loading
      volatilities[i] * 100          // Asset Daily Volatility %
    ];
  });
  
  // Standardize features
  const stdFeatures = standardizeFeatures(rawFeatures);
  
  // 2. Run DBSCAN
  const dbscanResults = runDBSCAN(stdFeatures, state.dbscanEps, state.dbscanMinSamples);
  
  // 3. Run KNN neighborhood scoring
  const knnScores = runKNNNeighborhoodScoring(dbscanResults.distances, state.knnK);
  
  // Compile assets list with metrics
  state.assets = ASSETS_CONFIG.map((conf, i) => {
    // Calculate Sharpe Ratio (assuming annualized Rf = 2.0%, Daily Rf = 0.008%)
    const dailyRf = 0.00008;
    const dailySharpe = volatilities[i] > 0 ? (avgReturns[i] - dailyRf) / volatilities[i] : 0;
    const annualizedSharpe = dailySharpe * Math.sqrt(252);
    
    // Count anomalies triggered
    let anomalyCount = 0;
    const pcaAnomaly = pcaResults.residuals[i] > state.pcaThreshold;
    const dbscanAnomaly = dbscanResults.labels[i] === -1;
    const knnAnomaly = knnScores[i] > state.knnThreshold;
    
    if (pcaAnomaly) anomalyCount++;
    if (dbscanAnomaly) anomalyCount++;
    if (knnAnomaly) anomalyCount++;
    
    let riskStatus = 'Safe';
    if (anomalyCount === 1) riskStatus = 'Warning';
    else if (anomalyCount >= 2) riskStatus = 'Critical';
    
    return {
      index: i,
      ticker: conf.ticker,
      name: conf.name,
      sector: conf.sector,
      desc: conf.desc,
      returns: returns[i],
      meanReturn: (avgReturns[i] * 100).toFixed(3), // daily %
      volatility: (volatilities[i] * 100).toFixed(2), // daily %
      sharpe: annualizedSharpe.toFixed(2),
      
      // Features
      pc1Loading: pcaResults.eigenvectors[0][i],
      pc2Loading: pcaResults.eigenvectors[1][i],
      
      // Algorithm outputs
      pcaResidual: pcaResults.residuals[i],
      pcaFlag: pcaAnomaly,
      
      dbscanCluster: dbscanResults.labels[i],
      dbscanFlag: dbscanAnomaly,
      
      knnScore: knnScores[i],
      knnFlag: knnAnomaly,
      
      anomalyCount: anomalyCount,
      riskStatus: riskStatus
    };
  });
  
  // Update portfolio-level states
  const totalAnomalies = state.assets.filter(a => a.riskStatus !== 'Safe').length;
  const portVol = volatilities.reduce((sum, v) => sum + v, 0) / N * 100;
  
  document.getElementById('total-anomalies-count').innerText = totalAnomalies;
  document.getElementById('portfolio-vol-value').innerText = portVol.toFixed(2) + '%';
  document.getElementById('pca-variance-pct').innerText = pcaResults.explainedVariancePct + '%';
  
  const healthVal = Math.max(0, 100 - (totalAnomalies * 8)).toFixed(1);
  const healthEl = document.getElementById('stat-health');
  healthEl.querySelector('.stat-value').innerText = healthEl.querySelector('.stat-indicator').innerText = '';
  
  if (healthVal >= 90) {
    healthEl.querySelector('.stat-value').className = 'stat-value text-emerald';
    healthEl.querySelector('.stat-value').innerText = healthVal + '%';
    healthEl.querySelector('.stat-indicator').className = 'stat-indicator normal';
    healthEl.querySelector('.stat-indicator').innerText = 'Stable';
  } else if (healthVal >= 75) {
    healthEl.querySelector('.stat-value').className = 'stat-value text-amber';
    healthEl.querySelector('.stat-value').innerText = healthVal + '%';
    healthEl.querySelector('.stat-indicator').className = 'stat-indicator warning';
    healthEl.querySelector('.stat-indicator').innerText = 'Elevated Risk';
  } else {
    healthEl.querySelector('.stat-value').className = 'stat-value text-rose';
    healthEl.querySelector('.stat-value').innerText = healthVal + '%';
    healthEl.querySelector('.stat-indicator').className = 'stat-indicator warning';
    healthEl.querySelector('.stat-indicator').innerText = 'Under Stress';
  }
  
  // Set regime descriptor based on variance and anomalies
  const regimeEl = document.getElementById('market-regime-value');
  if (totalAnomalies >= 4) {
    regimeEl.innerText = 'High Idiosyncratic Shift';
    regimeEl.className = 'stat-value text-rose';
  } else if (portVol > 1.4) {
    regimeEl.innerText = 'Systemic High-Vol';
    regimeEl.className = 'stat-value text-amber';
  } else {
    regimeEl.innerText = 'Systemic Low-Vol';
    regimeEl.className = 'stat-value text-emerald';
  }
  
  // Set Max Indicators in Card Footers
  const sortedByResidual = [...state.assets].sort((a, b) => b.pcaResidual - a.pcaResidual);
  document.getElementById('pca-top-residual').innerText = `${sortedByResidual[0].ticker} (${sortedByResidual[0].pcaResidual.toFixed(2)}%)`;
  
  const outliersCount = state.assets.filter(a => a.dbscanFlag).length;
  const numClusters = new Set(state.assets.map(a => a.dbscanCluster).filter(l => l !== -1)).size;
  document.getElementById('dbscan-cluster-count').innerText = `${numClusters} Clusters`;
  document.getElementById('dbscan-outliers-count').innerText = `${outliersCount} Outliers`;
  
  const sortedByKnn = [...state.assets].sort((a, b) => b.knnScore - a.knnScore);
  document.getElementById('knn-max-score').innerText = `${sortedByKnn[0].ticker} (${sortedByKnn[0].knnScore.toFixed(2)})`;
}

/**
 * Step 5: Dashboard Visualizations using Chart.js
 */
function initCharts() {
  // Chart.js Default styling tweaks for dark mode
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
  Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
  Chart.defaults.responsive = true;
  Chart.defaults.maintainAspectRatio = false;

  // 1. Chart 1: PCA Residual Scatter Plot
  const ctxPca = document.getElementById('chart-pca-residual').getContext('2d');
  state.charts.pcaResidual = new Chart(ctxPca, {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      scales: {
        x: { 
          title: { display: true, text: 'Daily Volatility (%)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.03)' }
        },
        y: { 
          title: { display: true, text: 'PCA Residual (Reconstruction Error %)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.03)' }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function(ctx) {
              const asset = state.assets[ctx.dataIndex];
              return `${asset.ticker} (${asset.name}): Vol=${asset.volatility}%, Residual=${asset.pcaResidual.toFixed(2)}%`;
            }
          }
        },
        legend: { display: false }
      },
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const idx = elements[0].index;
          selectAsset(state.assets[idx].index);
        }
      }
    }
  });

  // 2. Chart 2: DBSCAN Cluster Scatter Map
  const ctxDbscan = document.getElementById('chart-dbscan-clustering').getContext('2d');
  state.charts.dbscanCluster = new Chart(ctxDbscan, {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      scales: {
        x: { 
          title: { display: true, text: 'Factor 1 Exposure (PC1)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.03)' }
        },
        y: { 
          title: { display: true, text: 'Factor 2 Exposure (PC2)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.03)' }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function(ctx) {
              const asset = state.assets[ctx.dataIndex];
              const cName = asset.dbscanCluster === -1 ? 'Outlier (Noise)' : `Cluster ${asset.dbscanCluster}`;
              return `${asset.ticker}: PC1=${asset.pc1Loading.toFixed(2)}, PC2=${asset.pc2Loading.toFixed(2)} (${cName})`;
            }
          }
        },
        legend: { display: false }
      },
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const idx = elements[0].index;
          selectAsset(state.assets[idx].index);
        }
      }
    }
  });

  // 3. Chart 3: KNN Scores Bar Chart
  const ctxKnn = document.getElementById('chart-knn-scoring').getContext('2d');
  state.charts.knnScore = new Chart(ctxKnn, {
    type: 'bar',
    data: { labels: [], datasets: [] },
    options: {
      scales: {
        y: { 
          title: { display: true, text: 'KNN Isolation Score (Average 5-NN Distance)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.03)' }
        },
        x: {
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: false }
      },
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const idx = elements[0].index;
          // Chart is sorted by score, find actual asset index using the ticker label
          const ticker = state.charts.knnScore.data.labels[idx];
          const asset = state.assets.find(a => a.ticker === ticker);
          selectAsset(asset.index);
        }
      }
    }
  });

  // 4. Asset Detail Mini Chart (Line Chart of Returns)
  const ctxDetail = document.getElementById('chart-asset-returns').getContext('2d');
  state.charts.assetDetail = new Chart(ctxDetail, {
    type: 'line',
    data: {
      labels: Array(100).fill(0).map((_, i) => i + 1),
      datasets: [{
        label: 'Daily Returns %',
        data: [],
        borderColor: '#a78bfa',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        backgroundColor: 'rgba(139, 92, 246, 0.05)',
        tension: 0.1
      }]
    },
    options: {
      scales: {
        x: { display: false },
        y: { 
          ticks: { font: { size: 9 } },
          grid: { color: 'rgba(255,255,255,0.03)' }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function updateCharts() {
  const assets = state.assets;
  
  // --- 1. Update PCA Residual Scatter ---
  const normalData = [];
  const anomalyData = [];
  
  assets.forEach(a => {
    const pt = { x: parseFloat(a.volatility), y: a.pcaResidual };
    if (a.pcaFlag) {
      anomalyData.push(pt);
    } else {
      normalData.push(pt);
    }
  });
  
  state.charts.pcaResidual.data.datasets = [
    {
      label: 'Normal',
      data: normalData,
      backgroundColor: 'rgba(16, 185, 129, 0.7)',
      borderColor: '#10b981',
      pointRadius: 6,
      pointHoverRadius: 8
    },
    {
      label: 'Outlier',
      data: anomalyData,
      backgroundColor: 'rgba(244, 63, 94, 0.9)',
      borderColor: '#f43f5e',
      pointRadius: 8,
      pointStyle: 'rectRot', // Diamonds for outliers
      pointHoverRadius: 10
    }
  ];
  
  // Custom Annotation line for threshold in ChartJS
  // We draw a dotted line programmatically or by adding metadata
  state.charts.pcaResidual.options.plugins.annotation = {
    annotations: {
      line1: {
        type: 'line',
        yMin: state.pcaThreshold,
        yMax: state.pcaThreshold,
        borderColor: 'rgba(244, 63, 94, 0.5)',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          display: true,
          content: 'Risk Cutoff'
        }
      }
    }
  };
  state.charts.pcaResidual.update();

  // --- 2. Update DBSCAN Cluster Scatter ---
  // Color palette for clusters
  const clusterColors = [
    '#3b82f6', // Cluster 1 - Blue
    '#8b5cf6', // Cluster 2 - Purple
    '#06b6d4', // Cluster 3 - Cyan
    '#f59e0b', // Cluster 4 - Amber
    '#10b981', // Cluster 5 - Emerald
    '#ec4899', // Cluster 6 - Pink
    '#14b8a6', // Cluster 7 - Teal
  ];
  
  const dbscanDatasets = [];
  const outlierPts = [];
  
  // Group assets by cluster assignment
  const clusters = {};
  assets.forEach(a => {
    const pt = { x: a.pc1Loading, y: a.pc2Loading };
    if (a.dbscanCluster === -1) {
      outlierPts.push(pt);
    } else {
      if (!clusters[a.dbscanCluster]) clusters[a.dbscanCluster] = [];
      clusters[a.dbscanCluster].push(pt);
    }
  });
  
  // Add cluster datasets
  Object.keys(clusters).forEach((cId) => {
    const index = (parseInt(cId) - 1) % clusterColors.length;
    dbscanDatasets.push({
      label: `Cluster ${cId}`,
      data: clusters[cId],
      backgroundColor: clusterColors[index],
      pointRadius: 6,
      pointHoverRadius: 8
    });
  });
  
  // Add outlier dataset
  if (outlierPts.length > 0) {
    dbscanDatasets.push({
      label: 'DBSCAN Outliers',
      data: outlierPts,
      backgroundColor: '#f43f5e',
      borderColor: '#fda4af',
      borderWidth: 1.5,
      pointRadius: 8,
      pointHoverRadius: 10,
      pointStyle: 'rectRot' // diamond style
    });
  }
  
  state.charts.dbscanCluster.data.datasets = dbscanDatasets;
  state.charts.dbscanCluster.update();

  // --- 3. Update KNN Scores Bar Chart ---
  // Sort assets by KNN score descending
  const sortedKnn = [...assets].sort((a, b) => b.knnScore - a.knnScore);
  const labels = sortedKnn.map(a => a.ticker);
  const scores = sortedKnn.map(a => a.knnScore);
  const barColors = sortedKnn.map(a => a.knnFlag ? 'rgba(244, 63, 94, 0.8)' : 'rgba(16, 185, 129, 0.4)');
  const borderColors = sortedKnn.map(a => a.knnFlag ? '#f43f5e' : '#10b981');
  
  state.charts.knnScore.data.labels = labels;
  state.charts.knnScore.data.datasets = [{
    label: 'KNN Isolation Score',
    data: scores,
    backgroundColor: barColors,
    borderColor: borderColors,
    borderWidth: 1,
    borderRadius: 4
  }];
  
  state.charts.knnScore.update();
}

/**
 * Step 6: Render Risk Log Table & Bind UI Elements
 */
function renderAssetTable() {
  const tbody = document.getElementById('anomaly-table-body');
  tbody.innerHTML = '';
  
  state.assets.forEach((a) => {
    const tr = document.createElement('tr');
    tr.id = `row-asset-${a.ticker}`;
    if (a.index === state.selectedAssetIndex) {
      tr.className = 'selected-row';
    }
    
    // Setup risk status badge class
    let statusClass = 'status-ok';
    if (a.riskStatus === 'Warning') statusClass = 'status-warn';
    else if (a.riskStatus === 'Critical') statusClass = 'status-crit';
    
    const dbscanLabel = a.dbscanCluster === -1 ? 'Outlier (-1)' : `Cluster ${a.dbscanCluster}`;
    
    tr.innerHTML = `
      <td class="asset-ticker">${a.ticker}</td>
      <td>${a.sector}</td>
      <td class="mono ${a.pcaFlag ? 'text-rose' : ''}">${a.pcaResidual.toFixed(2)}%</td>
      <td class="${a.dbscanFlag ? 'text-rose' : ''}">${dbscanLabel}</td>
      <td class="mono ${a.knnFlag ? 'text-rose' : ''}">${a.knnScore.toFixed(2)}</td>
      <td><span class="status-pill ${statusClass}">${a.riskStatus}</span></td>
    `;
    
    tr.addEventListener('click', () => selectAsset(a.index));
    tbody.appendChild(tr);
  });
}

function selectAsset(index) {
  // Update state and styles
  const prevSelected = document.querySelector('.selected-row');
  if (prevSelected) prevSelected.classList.remove('selected-row');
  
  state.selectedAssetIndex = index;
  const asset = state.assets[index];
  
  const nextSelected = document.getElementById(`row-asset-${asset.ticker}`);
  if (nextSelected) nextSelected.classList.add('selected-row');
  
  // Open drawer
  const drawer = document.getElementById('asset-detail-drawer');
  drawer.classList.remove('closed');
  
  // Populate drawer statistics
  document.getElementById('detail-asset-sector').innerText = asset.sector;
  document.getElementById('detail-asset-name').innerText = asset.ticker;
  document.getElementById('detail-asset-desc').innerText = asset.name + ' - ' + asset.desc;
  
  // Find average return across daily return array and format
  const pctRet = parseFloat(asset.meanReturn);
  document.getElementById('detail-stat-return').innerText = (pctRet >= 0 ? '+' : '') + pctRet.toFixed(3) + '%';
  document.getElementById('detail-stat-vol').innerText = asset.volatility + '%';
  document.getElementById('detail-stat-sharpe').innerText = asset.sharpe;
  
  // Update metrics bars in drawer
  // PCA Residual
  const maxPcaResidualVal = Math.max(...state.assets.map(a => a.pcaResidual));
  const pcaPct = (asset.pcaResidual / (maxPcaResidualVal || 1.0)) * 100;
  document.getElementById('detail-metric-pca-val').innerText = asset.pcaResidual.toFixed(2) + '%';
  const pcaBar = document.getElementById('detail-metric-pca-bar');
  pcaBar.style.width = Math.min(100, Math.max(10, pcaPct)) + '%';
  
  const pcaStatusEl = document.getElementById('detail-metric-pca-status');
  if (asset.pcaFlag) {
    pcaStatusEl.innerText = 'High Residual';
    pcaStatusEl.className = 'metric-status status-danger';
    pcaBar.style.background = 'var(--color-rose)';
  } else {
    pcaStatusEl.innerText = 'Normal';
    pcaStatusEl.className = 'metric-status status-safe';
    pcaBar.style.background = 'var(--color-emerald)';
  }
  
  // DBSCAN Clustering
  const dbscanValEl = document.getElementById('detail-metric-dbscan-val');
  const dbscanStatusEl = document.getElementById('detail-metric-dbscan-status');
  const dbscanBar = document.getElementById('detail-metric-dbscan-bar');
  
  if (asset.dbscanFlag) {
    dbscanValEl.innerText = 'Noise (-1)';
    dbscanStatusEl.innerText = 'Outlier';
    dbscanStatusEl.className = 'metric-status status-danger';
    dbscanBar.style.width = '100%';
    dbscanBar.style.background = 'var(--color-rose)';
  } else {
    dbscanValEl.innerText = `Cluster ${asset.dbscanCluster}`;
    dbscanStatusEl.innerText = 'Correlated';
    dbscanStatusEl.className = 'metric-status status-safe';
    dbscanBar.style.width = '70%';
    dbscanBar.style.background = 'var(--color-blue)';
  }
  
  // KNN Score
  const maxKnnVal = Math.max(...state.assets.map(a => a.knnScore));
  const knnPct = (asset.knnScore / (maxKnnVal || 1.0)) * 100;
  document.getElementById('detail-metric-knn-val').innerText = asset.knnScore.toFixed(2);
  const knnBar = document.getElementById('detail-metric-knn-bar');
  knnBar.style.width = Math.min(100, Math.max(10, knnPct)) + '%';
  
  const knnStatusEl = document.getElementById('detail-metric-knn-status');
  if (asset.knnFlag) {
    knnStatusEl.innerText = 'Isolated';
    knnStatusEl.className = 'metric-status status-danger';
    knnBar.style.background = 'var(--color-rose)';
  } else {
    knnStatusEl.innerText = 'Normal';
    knnStatusEl.className = 'metric-status status-safe';
    knnBar.style.background = 'var(--color-emerald)';
  }
  
  // Risk Advisory Text Box
  const advisoryBox = document.getElementById('detail-advisory-box');
  const advisoryIcon = document.getElementById('advisory-icon');
  const advisoryText = document.getElementById('detail-advisory-text');
  
  if (asset.anomalyCount >= 2) {
    advisoryBox.className = 'detail-section advisory-box alert-critical';
    advisoryIcon.setAttribute('data-lucide', 'shield-x');
    let report = `CRITICAL EXPOSURE DETECTED: This asset is flagged by multiple algorithms. `;
    if (asset.pcaFlag) report += `It exhibits high idiosyncratic returns (PCA Residual of ${asset.pcaResidual.toFixed(2)}%), indicating corporate decoupling. `;
    if (asset.dbscanFlag) report += `DBSCAN isolates it as a style-drift outlier. `;
    if (asset.knnFlag) report += `It has an extremely high KNN Isolation score (${asset.knnScore.toFixed(2)}), making it difficult to proxy-hedge. `;
    report += `ACTION: Recommend scaling down exposure or purchasing tail-risk options.`;
    advisoryText.innerText = report;
  } else if (asset.anomalyCount === 1) {
    advisoryBox.className = 'detail-section advisory-box alert-warning';
    advisoryIcon.setAttribute('data-lucide', 'shield-alert');
    let report = `ELEVATED RISK FLAG: `;
    if (asset.pcaFlag) report += `High idiosyncratic variance detected. The asset is moving independently of systemic market factors.`;
    else if (asset.dbscanFlag) report += `The asset does not cluster with peers, indicating anomalous exposure.`;
    else if (asset.knnFlag) report += `Isolated asset behavior. Closest 5 peer assets are far in return-risk space.`;
    report += ` ACTION: Monitor sector correlation trends and verify if event risk is transient.`;
    advisoryText.innerText = report;
  } else {
    advisoryBox.className = 'detail-section advisory-box alert-normal';
    advisoryIcon.setAttribute('data-lucide', 'shield-check');
    advisoryText.innerText = `NORMAL OPERATION: Asset is behaving within normal parameter boundaries. Co-movements are well represented by systemic components. Standard portfolio allocation applies.`;
  }
  
  // Update Lucide icon in drawer
  lucide.createIcons();
  
  // Update Mini Line Chart
  state.charts.assetDetail.data.datasets[0].data = asset.returns.map(r => r * 100);
  state.charts.assetDetail.update();
}

/**
 * Setup Event Listeners and Initializers
 */
function setupUIListeners() {
  // 1. PCA Components Slider
  const pcaSlider = document.getElementById('pca-components');
  pcaSlider.addEventListener('input', (e) => {
    state.pcaComponents = parseInt(e.target.value);
    document.getElementById('pca-components-val').innerText = state.pcaComponents;
    runPipelineAndRender();
  });

  // 2. DBSCAN Epsilon Slider
  const epsSlider = document.getElementById('dbscan-eps');
  epsSlider.addEventListener('input', (e) => {
    state.dbscanEps = parseFloat(e.target.value);
    document.getElementById('dbscan-eps-val').innerText = state.dbscanEps;
    runPipelineAndRender();
  });

  // 3. DBSCAN Min Samples Slider
  const minSamplesSlider = document.getElementById('dbscan-min-samples');
  minSamplesSlider.addEventListener('input', (e) => {
    state.dbscanMinSamples = parseInt(e.target.value);
    document.getElementById('dbscan-min-samples-val').innerText = state.dbscanMinSamples;
    runPipelineAndRender();
  });

  // 4. KNN Threshold Slider
  const sensitivitySlider = document.getElementById('anomaly-sensitivity');
  sensitivitySlider.addEventListener('input', (e) => {
    state.knnThreshold = parseFloat(e.target.value);
    document.getElementById('anomaly-sensitivity-val').innerText = state.knnThreshold;
    runPipelineAndRender();
  });

  // Populate Target Asset Dropdown in Simulator
  const shockTargetSelect = document.getElementById('shock-target');
  ASSETS_CONFIG.forEach((asset, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    opt.innerText = `${asset.ticker} - ${asset.sector}`;
    shockTargetSelect.appendChild(opt);
  });

  // 5. Inject Shock Event
  document.getElementById('btn-inject-shock').addEventListener('click', () => {
    const targetIdx = parseInt(document.getElementById('shock-target').value);
    const shockType = document.getElementById('shock-type').value;
    injectMarketShock(targetIdx, shockType);
  });

  // 6. Reset Portfolio Data
  document.getElementById('btn-reset-data').addEventListener('click', () => {
    state.returns = generatePortfolioReturns(125);
    runPipelineAndRender();
    // Re-select active asset
    selectAsset(state.selectedAssetIndex);
  });

  // 7. Close Detail Drawer
  document.getElementById('btn-close-detail').addEventListener('click', () => {
    document.getElementById('asset-detail-drawer').classList.add('closed');
    const row = document.querySelector('.selected-row');
    if (row) row.classList.remove('selected-row');
  });
}

/**
 * Shock Simulator
 * Alters the underlying returns matrix to simulate financial stress events.
 */
function injectMarketShock(assetIndex, shockType) {
  const T = 100;
  const N = 30;
  
  if (shockType === 'idiosyncratic') {
    // Inject massive idiosyncratic shock (e.g. CEO resignation, fraud discovery) on final day
    // And increase overall idiosyncratic volatility
    const randObj = { seed: 99 };
    for (let t = 80; t < T; t++) {
      state.returns[assetIndex][t] += randomNormal(randObj) * 0.05; // High volatility spikes
    }
    state.returns[assetIndex][T-1] = 0.18; // Massive +18% shock return
    
  } else if (shockType === 'decoupling') {
    // Decoupling from systemic market/sector factors - pure noise for the last 30 days
    const randObj = { seed: 444 };
    for (let t = 70; t < T; t++) {
      state.returns[assetIndex][t] = randomNormal(randObj) * 0.035; // Volatile decoupled noise
    }
    
  } else if (shockType === 'sector-meltdown') {
    // Get sector of the target asset
    const targetAsset = state.assets[assetIndex];
    const sector = targetAsset.sector;
    
    // Apply systematic collapse to all assets in that sector (-3% return per day for last 5 days)
    state.assets.forEach(asset => {
      if (asset.sector === sector) {
        for (let t = 95; t < T; t++) {
          state.returns[asset.index][t] -= 0.035;
        }
      }
    });
    
  } else if (shockType === 'systemic-crash') {
    // Flash Crash: Massive systemic selloff across ALL assets on final day
    const randObj = { seed: 50 };
    for (let i = 0; i < N; i++) {
      state.returns[i][T-1] = -0.06 - (lcgRandom(randObj) * 0.04); // -6% to -10% drop
    }
  }

  // Recalculate everything and render
  runPipelineAndRender();
  // Select the shocked asset
  selectAsset(assetIndex);
}

function runPipelineAndRender() {
  processPortfolioData();
  updateCharts();
  renderAssetTable();
}

// Initialize Application
window.addEventListener('DOMContentLoaded', () => {
  // Load Icons
  lucide.createIcons();
  
  // 1. Generate Returns
  state.returns = generatePortfolioReturns(state.randomSeed);
  
  // 2. Init Charts
  initCharts();
  
  // 3. Bind UI Controls
  setupUIListeners();
  
  // 4. Run calculations & render
  runPipelineAndRender();
  
  // 5. Select first asset by default
  selectAsset(0);
});
