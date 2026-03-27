Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add support for machine-operation event markers that may change deposition behavior.

Goal:
In reality, machine interventions can create behavior shifts, so the model should be able to recognize these events and adjust interpretation of before/after data.

Implement the following:
1. Add a data structure and UI support for “machine events” or “process events”
2. Each event should include:
   - date/time or run position
   - selected material or global scope depending on design
   - event type / label
   - optional notes
3. Allow the user to mark an event point between deposition records
4. Use event markers in modeling and analysis:
   - visualize event lines on lifetime or time plots
   - optionally split evaluation into before-event and after-event
   - optionally add event-segment features to the model
5. Add comparison views:
   - model performance before event
   - model performance after event
   - whether drift increased
6. Do not force full event-aware modeling immediately; start with event tagging + segmentation-aware evaluation, then expose extension points for future modeling

Implementation guidance:
- Keep event model general enough for chamber operations, maintenance, calibration, etc.
- Make event markers visible in plots and evaluation summaries
- Add tests for event insertion and segmented evaluation logic

Output:
- Add machine-event marking support
- Make before/after performance visible in the app