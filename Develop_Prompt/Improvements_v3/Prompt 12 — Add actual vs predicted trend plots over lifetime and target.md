Implement trend-style plots similar to the existing E.trend idea, but for model validation. Do not change unrelated behavior.

Goal:
Visualize how predicted results differ from actual measured results over lifetime or target order.

Required plots:
1. Actual vs Predicted target value over lifetime
2. In-spec actual vs predicted over lifetime
3. Optional smoothed/fitted trend lines connecting in-spec points

Requirements:
1. Reuse the backtest/evaluation result objects from earlier prompts.
2. Let the user choose:
   - x-axis mode: lifetime / target index / run order
   - y-axis target: RS / thickness / RSU / in-spec probability if supported
3. Show actual measured series and predicted series together.
4. For in-spec style view:
   - actual in-spec points
   - predicted in-spec or predicted pass probability
5. Add optional trend line / smoothing for visual guidance, but keep raw points visible.
6. Make the UI style similar to E.trend where practical.

Acceptance criteria:
- I can compare actual and predicted evolution over lifetime.
- I can inspect whether the model follows the same in-spec trend direction as the real process.