"""my-team/team.py - a complete tactical team."""

from soccer import (
    TeamAction, TeamController, clamp, closest_to_ball, direction, distance, closest_point_on_segment
)

def predict_ball(obs, ticks):
    """Where the ball will be, under the drag the engine applies."""
    dt = 1.0 / obs.field.simulation_hz
    x, y = obs.ball.position
    vx, vy = obs.ball.velocity
    for _ in range(ticks):
        x, y = x + vx * dt, y + vy * dt
        vx, vy = vx * obs.field.ball_friction, vy * obs.field.ball_friction
    return x, y

def lane_room(obs, a, b):
    """How much room the lane a->b has: the distance from the nearest
    opponent to that segment. `closest_point_on_segment(point, seg_a, seg_b)`
    wants the *opponent* as `point` and the lane's two ends as `seg_a`/`seg_b`
    - get the argument order backwards and every lane looks blocked (or, just
    as often, wide open when it isn't)."""
    room = float('inf')
    for opp in obs.opponents:
        nearest_point_on_lane = closest_point_on_segment(opp.position, a, b)
        gap = distance(opp.position, nearest_point_on_lane)
        if gap < room:
            room = gap
    return room

class MyTeam(TeamController):
    name = "my_team"
    version = "5"

    def initial_formation(self, field):
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
        # Use velocity == (0.0, 0.0) and distance to (0,0) < 2.0 to reliably detect waiting kickoffs.
        if distance(obs.ball.position, (0.0, 0.0)) < 2.0 and obs.ball.velocity == (0.0, 0.0) and obs.ball.controlling_team == 1:
            return self.hold_shape_only(obs)

        if closest_to_ball(obs):
            return self.on_the_ball(obs)
        return self.off_the_ball(obs)

    def hold_shape_only(self, obs):
        actions = TeamAction()
        for player in obs.my_players:
            if player.id == 0:
                mouth = obs.field.goal_width / 2
                spot = (obs.my_goal[0] + 2.0, clamp(obs.ball.position[1], -mouth, mouth))
                actions.move(player.id, direction(player.position, spot))
            else:
                base_x, base_y = self.initial_formation(obs.field)[player.id]
                actions.move(player.id, direction(player.position, (base_x, base_y)))
        return actions

    def find_best_pass(self, obs, passer):
        """Returns (best_forward, best_backward). Kept separate rather than
        collapsed into one "best pass" - a shot decision needs to know
        whether a *forward* option exists without a trailing, always-open
        backward option masking that there wasn't one."""
        best_forward = None
        best_forward_score = -float('inf')

        best_backward = None
        best_backward_room = -1.0

        for teammate in obs.my_players:
            if teammate.id == passer.id:
                continue
            if teammate.id == 0:
                continue

            # Room the lane from passer -> teammate actually has: the
            # distance from the nearest opponent to that segment, not to
            # some unrelated segment.
            min_room = lane_room(obs, passer.position, teammate.position)

            # Forward pass
            if teammate.position[0] > passer.position[0]:
                if min_room > 2.0:
                    progress = teammate.position[0] - passer.position[0]
                    score = min_room * progress
                    if score > best_forward_score:
                        best_forward_score = score
                        best_forward = teammate
            # Backward / sideways pass fallback
            else:
                if min_room > 2.0 and min_room > best_backward_room:
                    best_backward_room = min_room
                    best_backward = teammate

        return best_forward, best_backward

    def find_best_shot(self, obs, shooter):
        """Aim along whichever line to the goal mouth has the most room,
        instead of always shooting straight at the centre of the goal.
        Returns (target, room) so callers can judge how clean the look is."""
        mouth = obs.field.goal_width / 2
        gx = obs.opponent_goal[0]
        best_target = (gx, 0.0)
        best_room = -float('inf')
        for y in [0.0, mouth * 0.9, -mouth * 0.9, mouth * 0.45, -mouth * 0.45]:
            target = (gx, y)
            room = lane_room(obs, shooter.position, target)
            if room > best_room:
                best_room = room
                best_target = target
        return best_target, best_room

    def get_chaser_intercept(self, obs, player_pos):
        predicted_pos = predict_ball(obs, 40)
        return closest_point_on_segment(player_pos, obs.ball.position, predicted_pos)

    def get_lead_pass_target(self, obs, passer, target_player):
        target_pos = target_player.position
        for _ in range(2):
            dist = distance(passer.position, target_pos)
            power = clamp(dist / 33.0, 0.3, 1.0)
            ticks = obs.ticks_to_cover(dist, power)
            dt = ticks / obs.field.simulation_hz
            # Assume receiver sprints towards opponent goal at max speed
            target_pos = (target_player.position[0] + (obs.field.max_speed * dt * 0.8), target_player.position[1])
            # Clamp to field boundaries
            target_pos = (clamp(target_pos[0], -obs.field.width/2, obs.field.width/2),
                          clamp(target_pos[1], -obs.field.height/2, obs.field.height/2))
        return target_pos

    def assign_markers(self, obs, markers):
        """Pair each available marker with a distinct, dangerous opponent -
        goalside of them - instead of letting several markers latch onto the
        same opponent while someone else in space goes untracked."""
        if not markers:
            return {}
        opponent_keeper_id = min((o.id for o in obs.opponents), default=None)
        targets = [o for o in obs.opponents if o.id != opponent_keeper_id]
        # Mark the most advanced (most dangerous) opponents first.
        targets.sort(key=lambda o: -o.position[0])

        assignment = {}
        remaining_markers = list(markers)
        for opp in targets:
            if not remaining_markers:
                break
            nearest = min(remaining_markers, key=lambda p: distance(p.position, opp.position))
            assignment[nearest.id] = opp
            remaining_markers.remove(nearest)
        return assignment

    def on_the_ball(self, obs):
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
                if obs.can_kick(player.id):
                    gap = distance(player.position, obs.opponent_goal)
                    forward_target, backward_target = self.find_best_pass(obs, player)
                    shot_target, shot_room = (
                        self.find_best_shot(obs, player) if gap < 45.0 else (None, -1.0)
                    )

                    # Shooting must be able to win against a "safe" backward
                    # pass, or a trailing team-mate who is always open ends up
                    # vetoing every shot - which is what silently strangled
                    # this team's scoring before. Close in, take a reasonably
                    # clean look over anything else; further out, still take
                    # it if there is no better forward option on.
                    close_range_look = gap < 25.0 and shot_room > 2.5
                    no_better_forward_option = forward_target is None and shot_room > 1.5
                    take_shot = shot_target is not None and (close_range_look or no_better_forward_option)

                    if take_shot:
                        actions.kick(
                            player.id,
                            direction(player.position, shot_target),
                            kick_power=1.0
                        )
                    elif forward_target or backward_target:
                        target = forward_target or backward_target
                        lead_target = self.get_lead_pass_target(obs, player, target)
                        pass_gap = distance(player.position, lead_target)
                        actions.kick(
                            player.id,
                            direction(player.position, lead_target),
                            kick_power=clamp(pass_gap / 33.0, 0.3, 1.0)
                        )
                    elif gap < 45.0:
                        actions.kick(
                            player.id,
                            direction(player.position, shot_target),
                            kick_power=1.0
                        )
                    else:
                        # Step 1: Real dribble state
                        # Light touch to close ground slightly faster than dribbling, instead of punting
                        actions.kick(
                            player.id,
                            direction(player.position, obs.opponent_goal),
                            kick_power=0.15,
                            movement=direction(player.position, obs.opponent_goal)
                        )
                else:
                    intercept_spot = self.get_chaser_intercept(obs, player.position)
                    actions.move(
                        player.id,
                        direction(player.position, intercept_spot)
                    )
            else:
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

        marker_candidates = [
            p for p in obs.my_players
            if p.id != 0 and not (p.id == chaser.id or (p.id == 0 and keeper_is_chaser))
        ]
        assignment = self.assign_markers(obs, marker_candidates)

        for player in obs.my_players:
            if player.id == 0 and not keeper_is_chaser:
                mouth = obs.field.goal_width / 2
                spot = (obs.my_goal[0] + 2.0, clamp(obs.ball.position[1], -mouth, mouth))
                actions.move(player.id, direction(player.position, spot))

            elif player.id == chaser.id or (player.id == 0 and keeper_is_chaser):
                intercept_spot = self.get_chaser_intercept(obs, player.position)
                actions.move(
                    player.id, direction(player.position, intercept_spot)
                )

            else:
                them = assignment.get(player.id)
                if them is not None and them.position[0] < 10.0:
                    # Goalside: on the line between the opponent and the goal
                    # we defend, not just shifted along x.
                    to_goal = direction(them.position, obs.my_goal)
                    spot = (them.position[0] + to_goal[0] * 3.0,
                            them.position[1] + to_goal[1] * 3.0)
                    actions.move(player.id, direction(player.position, spot))
                else:
                    base_x, base_y = self.initial_formation(obs.field)[player.id]
                    shift_x = obs.ball.position[0] * 0.5
                    spot = (base_x + shift_x, base_y)
                    actions.move(player.id, direction(player.position, spot))

        return actions