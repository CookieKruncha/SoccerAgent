"""my-team/team.py - a complete tactical team."""

from soccer import (
    TeamAction, TeamController, clamp, closest_to_ball, direction, distance, closest_point_on_segment
)


class MyTeam(TeamController):
    name = "my_team"
    version = "3"

    def initial_formation(self, field):
        # Priority 8: Design kickoff formation around winning the kickoff.
        # Place the forward exactly on the edge of the centre circle to win first touch.
        forward_x = -field.centre_circle_radius - 0.1
        return [
            (-48.0, 0.0),            # slot 0 (keeper)
            (-20.0, -15.0),          # slot 1 (left back)
            (-20.0, 15.0),           # slot 2 (right back)
            (forward_x - 5.0, -10.0),# slot 3 (left forward)
            (forward_x, 0.0),        # slot 4 (central forward, kickoff taker)
        ]

    def act(self, obs):
        # Priority 9: Avoid fouls.
        # If ball is exactly at (0, 0) and controlling_team == 1, it's their kickoff.
        # We must NOT enter the centre circle or we get a foul. Hold shape instead.
        if obs.ball.position == (0.0, 0.0) and obs.ball.controlling_team == 1:
            return self.hold_shape_only(obs)

        # Normal play
        if closest_to_ball(obs):
            return self.on_the_ball(obs)
        return self.off_the_ball(obs)

    def hold_shape_only(self, obs):
        actions = TeamAction()
        for player in obs.my_players:
            if player.id == 0:
                # Keeper logic (Priority 2)
                mouth = obs.field.goal_width / 2
                spot = (obs.my_goal[0] + 2.0, clamp(obs.ball.position[1], -mouth, mouth))
                actions.move(player.id, direction(player.position, spot))
            else:
                # Hold shape based on initial formation
                base_x, base_y = self.initial_formation(obs.field)[player.id]
                actions.move(player.id, direction(player.position, (base_x, base_y)))
        return actions

    def find_best_pass(self, obs, passer):
        # Priority 3: Pass instead of always shooting
        best_teammate = None
        best_room = -1.0
        
        for teammate in obs.my_players:
            if teammate.id == passer.id: 
                continue
            if teammate.id == 0: 
                continue # Prefer not to pass to keeper
            
            # Prefer forward passes: teammate should be ahead of passer
            if teammate.position[0] <= passer.position[0]: 
                continue
            
            # Check lane room
            min_room = float('inf')
            for opp in obs.opponents:
                opp_dist_to_lane = distance(
                    opp.position, 
                    closest_point_on_segment(passer.position, teammate.position, opp.position)
                )
                if opp_dist_to_lane < min_room:
                    min_room = opp_dist_to_lane
            
            # If the lane has more than 2.0 units of room, it's a viable pass
            if min_room > 2.0 and min_room > best_room:
                best_room = min_room
                best_teammate = teammate
                
        return best_teammate

    def on_the_ball(self, obs):
        actions = TeamAction()
        
        # Priority 2: Keeper rushes if ball is deep in our half and keeper is closest
        keeper_is_chaser = False
        chaser = obs.closest_my_player_to(obs.ball.position)
        if chaser.id == 0 and obs.ball.position[0] < -35.0:
            keeper_is_chaser = True

        for player in obs.my_players:
            if player.id == 0 and not keeper_is_chaser:
                mouth = obs.field.goal_width / 2
                spot = (obs.my_goal[0] + 2.0, clamp(obs.ball.position[1], -mouth, mouth))
                actions.move(player.id, direction(player.position, spot))

            elif player.id == chaser.id or (player.id == 0 and keeper_is_chaser):
                if obs.can_kick(player.id):
                    target = self.find_best_pass(obs, player)
                    if target:
                        # Pass it!
                        gap = distance(player.position, target.position)
                        # Priority 4: Scale kick power for passes
                        actions.kick(
                            player.id,
                            direction(player.position, target.position),
                            kick_power=clamp(gap / 33.0, 0.3, 1.0)
                        )
                    else:
                        # Shoot or dribble
                        gap = distance(player.position, obs.opponent_goal)
                        if gap < 45.0: 
                            # In shooting range
                            actions.kick(
                                player.id,
                                direction(player.position, obs.opponent_goal),
                                kick_power=1.0
                            )
                        else:
                            # Dribble forward
                            actions.kick(
                                player.id,
                                direction(player.position, obs.opponent_goal),
                                kick_power=0.4,
                                movement=direction(player.position, obs.opponent_goal)
                            )
                else:
                    # Priority 7 (Cooldown): continue moving toward the ball/goal
                    actions.move(
                        player.id,
                        direction(player.position, obs.ball.position)
                    )
            else:
                # Shape holding for other players
                base_x, base_y = self.initial_formation(obs.field)[player.id]
                shift_x = obs.ball.position[0] * 0.5 
                spot = (base_x + shift_x, base_y)
                actions.move(player.id, direction(player.position, spot))

        return actions

    def off_the_ball(self, obs):
        actions = TeamAction()
        
        keeper_is_chaser = False
        chaser = obs.closest_my_player_to(obs.ball.position)
        if chaser.id == 0 and obs.ball.position[0] < -35.0:
            keeper_is_chaser = True

        for player in obs.my_players:
            if player.id == 0 and not keeper_is_chaser:
                mouth = obs.field.goal_width / 2
                spot = (obs.my_goal[0] + 2.0, clamp(obs.ball.position[1], -mouth, mouth))
                actions.move(player.id, direction(player.position, spot))

            elif player.id == chaser.id or (player.id == 0 and keeper_is_chaser):
                actions.move(
                    player.id, direction(player.position, obs.ball.position)
                )

            else:
                # Priority 5: Goalside marking
                them = obs.closest_opponent_to(player.position)
                if them is not None and them.position[0] < 10.0: 
                    # Only mark aggressively in our half/midfield.
                    # Stand exactly between the opponent and our goal.
                    spot = (them.position[0] - 3.0, them.position[1])
                    actions.move(player.id, direction(player.position, spot))
                else:
                    # Hold shape if opponent is far
                    base_x, base_y = self.initial_formation(obs.field)[player.id]
                    shift_x = obs.ball.position[0] * 0.5 
                    spot = (base_x + shift_x, base_y)
                    actions.move(player.id, direction(player.position, spot))

        return actions