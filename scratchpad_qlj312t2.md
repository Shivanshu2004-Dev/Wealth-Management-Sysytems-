# Task: Verify Wealth Management Portfolio Risk Monitoring Dashboard

## Checklist
- [ ] Open http://localhost:8000 and verify page loads
- [ ] Verify title "AURA RiskMonitor"
- [ ] Confirm three charts are rendered
- [ ] Check Risk Classification table for 'CRM', 'WFC', 'OXY' (Warning/Critical)
- [ ] Click 'CRM' row and verify detail drawer opens, shows details, advisory, mini history chart
- [ ] Inject stress event on AAPL: 'Idiosyncratic Shock (+7σ Jump)'
- [ ] Verify AAPL status moves to Critical/Warning and PCA residual spikes
- [ ] Capture screenshots at major steps (load, selection, post-shock)
- [ ] Check console logs for errors
