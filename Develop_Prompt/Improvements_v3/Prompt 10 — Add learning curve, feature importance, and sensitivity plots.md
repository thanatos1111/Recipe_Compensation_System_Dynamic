Add model understanding plots that help determine whether data is sufficient and which parameters matter. Do not change unrelated behavior.

Goal:
Show whether performance improves with more data and which process parameters drive the model.

Required views:
1. Learning curve
2. Feature importance
3. Parameter sensitivity / partial dependence style plot for selected features

Requirements:
1. Learning curve:
   - x = training sample count or fraction
   - y = train metric and validation metric
   - support at least MAE or RMSE
2. Feature importance:
   - for supported models, show ranked importance bar chart
   - if a model does not support native importance, handle gracefully
3. Sensitivity:
   - let user choose a feature
   - plot predicted target response as that feature changes while others are held fixed or representative
4. Add clear warnings that sensitivity/importance are model-based interpretations, not physical truth.
5. Keep the implementation compatible with the benchmark-selected winner model flow.

Acceptance criteria:
- I can see whether more data still improves the model.
- I can see which parameters are most influential.
- I can inspect how predicted outcome changes with one selected parameter.