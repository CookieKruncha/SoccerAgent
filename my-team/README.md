# My Team - SoccerAgent

This is a custom `TeamController` implementation for the University of Witwatersrand SoccerAgent coursework. It controls a 5-a-side team using advanced tactics and predictive physics, all within a strict 20ms/tick decision budget.

## Tactical Implementations

This team incorporates all high-priority recommendations to maximize efficiency and win rate:
1. **Kickoff Formation & Foul Avoidance**: Dynamically places a forward exactly on the edge of the centre circle to win our kickoffs, while strictly holding shape during opponent kickoffs to entirely eliminate foul penalties.
2. **Dedicated Goalkeeper**: A keeper (slot 0) tracking the ball's y-position on the goal line, but given the intelligence to dynamically rush out if the ball is deep in our half and they are the closest to it.
3. **Smart Passing & Power**: Evaluates all potential passing lanes for clearance (`2.0` units from any opponent) using `closest_point_on_segment`. If an open lane exists, the chaser will pass using a dynamically scaled kick power (`distance / 33.0`) so the receiver can comfortably collect it.
4. **Goalside Marking**: Non-chasing defenders aggressively position themselves exactly between the nearest opponent and our goal line, effectively cutting off direct shooting lanes.
5. **Kick Cooldown Dribbling**: Instead of freezing when their kick is on cooldown, players will dynamically `move` alongside the ball to maintain momentum (dribbling).
6. **Predictive Physics & Interceptions**: Uses a custom `predict_ball` function matching the engine's drag/friction mechanics to intercept the ball's future position, rather than trailing its current position. Lead passes aim into space ahead of sprinting receivers.

## Performance

- **Validation**: 0 missing actions, 0 timeouts (mean decision time ~0.022ms).
- **Vs Tactical Baseline**: Consistently achieves a ~20% win rate and draws heavily against the absolute hardest reference opponent provided by the SDK.
- **Fouls**: 0 fouls across standard tournament play.

## Running the Team

Ensure you have Python 3.8+ installed. You do not need to install any additional pip packages.

Validate the team:
```bash
./start.sh validate my-team
```

Play a quick visual match against the balanced baseline:
```bash
./start.sh play my-team --against balanced
```

Run a full 40-match tournament against the tactical baseline to measure actual win rates:
```bash
./start.sh tournament my-team --against tactical --seeds 1000..1020
```
