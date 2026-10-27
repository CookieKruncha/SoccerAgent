import math
from itertools import permutations

from soccer import (
    PlayerAction,
    TeamAction,
    TeamController,
    clamp,
    closest_point_on_segment,
    closest_to_ball,
    direction,
    distance,
)

def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return (0.0, 0.0)
    return (dx / d, dy / d)

def predict_ball_path(obs, ticks=40):
    """Where the ball will travel, including pitch boundary reflections."""
    dt = 1.0 / obs.field.simulation_hz
    fric = obs.field.ball_friction
    hx = obs.field.width / 2.0 - obs.field.ball_radius
    hy = obs.field.height / 2.0 - obs.field.ball_radius
    restit = getattr(obs.field, 'bounce_restitution', 0.75)

    x, y = obs.ball.position
    vx, vy = obs.ball.velocity
    path = []
    for _ in range(ticks):
        x += vx * dt
        y += vy * dt
        vx *= fric
        vy *= fric
        if x > hx:
            x = hx
            vx = -vx * restit
        elif x < -hx:
            x = -hx
            vx = -vx * restit
        if y > hy:
            y = hy
            vy = -vy * restit
        elif y < -hy:
            y = -hy
            vy = -vy * restit
        path.append((x, y))
    return path

def arrival_time(player_pos, player_vel, target_pos, reach, vmax, accel):
    """Kinematic travel time accounting for drift, braking, turn-cost, and acceleration."""
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    dist = math.hypot(dx, dy) - reach
    if dist <= 0.0:
        return 0.0

    inv_d = 1.0 / (dist + reach)
    dir_x = dx * inv_d
    dir_y = dy * inv_d
    v_towards = player_vel[0] * dir_x + player_vel[1] * dir_y

    if v_towards < 0.0:
        t_stop = -v_towards / accel
        dist_drift = 0.5 * (-v_towards) * t_stop
        d_total = dist + dist_drift
        t_accel = vmax / accel
        d_accel = 0.5 * accel * t_accel * t_accel
        if d_total <= d_accel:
            return t_stop + math.sqrt(max(0.0, 2.0 * d_total / accel))
        return t_stop + t_accel + (d_total - d_accel) / vmax

    v0 = min(v_towards, vmax)
    t_accel = (vmax - v0) / accel
    d_accel = v0 * t_accel + 0.5 * accel * t_accel * t_accel
    if dist <= d_accel:
        return (-v0 + math.sqrt(max(0.0, v0 * v0 + 2.0 * accel * dist))) / accel
    return t_accel + (dist - d_accel) / vmax

def lane_room(obs, a, b):
    room = float('inf')
    for opp in obs.opponents:
        nearest_point_on_lane = closest_point_on_segment(opp.position, a, b)
        gap = distance(opp.position, nearest_point_on_lane)
        if gap < room:
            room = gap
    return room

class MyTeam(TeamController):
    name = "my_team"
    version = "12"

    FORWARD_IDS = (3, 4)
    CHASER_HYSTERESIS = 0.15

    SHOOT_RANGE = 36.0
    SHOT_GATE_RANGE = 20.0
    JOCKEY_BALL_X = 5.0
    JOCKEY_GAP = 1.6
    GUARD_DIST = 8.0
    GUARD_STICKY = 4.0
    MARK_STICKY = 3.0
    MARK_MAX_X = 18.0
    CARRIER_RADIUS = 8.0
    MODE_DWELL_TICKS = 5

    def __init__(self):
        super().__init__()
        self._reset_state()
        self._last_tick = -1

    def _reset_state(self):
        self._chaser_id = None
        self._path = []
        self._icpt = {}
        self._we_scored = False
        self._seen_centre = False
        self._marks_prev = {}
        self._guard_prev = None
        self._sup_prev = {}
        self._mode_attacking = False
        self._mode_tick = 0
        self._last_team = 0
        self._shot_margin = 0.0

    def initial_formation(self, field):
        forward_x = -field.centre_circle_radius - 1.6
        return [
            (-48.0, 0.0),            # slot 0 (keeper)
            (-20.0, -15.0),          # slot 1 (left back)
            (-20.0, 15.0),           # slot 2 (right back)
            (forward_x - 5.0, -10.0),# slot 3 (left forward)
            (forward_x, 0.0),        # slot 4 (central forward, kickoff taker)
        ]

    def _go(self, pos, target, arrive=2.5):
        """Unit-length run towards target, throttled down inside `arrive` units."""
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        d = math.hypot(dx, dy)
        if d < 0.2:
            return (0.0, 0.0)
        s = 1.0 if d >= arrive else d / arrive
        return (dx / d * s, dy / d * s)

    def player_intercept(self, obs, player):
        f = obs.field
        reach = f.kick_range * 0.8
        vel = player.velocity
        vmax = f.max_speed
        accel = f.acceleration
        dt = 1.0 / f.simulation_hz

        holder = obs.ball.controlling_player
        if holder is not None and holder >= len(obs.my_players):
            opp = next((o for o in obs.opponents if o.id == holder), None)
            if opp is not None:
                opp_v = opp.velocity
                t_est = distance(player.position, opp.position) / max(vmax, 1e-4)
                lead_pt = (
                    clamp(opp.position[0] + opp_v[0] * min(t_est, 1.2), -f.width / 2.0 + 1.0, f.width / 2.0 - 1.0),
                    clamp(opp.position[1] + opp_v[1] * min(t_est, 1.2), -f.height / 2.0 + 1.0, f.height / 2.0 - 1.0),
                )
                t_me = arrival_time(player.position, vel, lead_pt, reach, vmax, accel)
                return t_me, lead_pt

        t_ball = 0.0
        pt = obs.ball.position
        for pt in self._path:
            t_ball += dt
            d = distance(player.position, pt) - reach
            if d > vmax * t_ball:
                continue
            t_me = arrival_time(player.position, vel, pt, reach, vmax, accel)
            if t_me <= t_ball + 0.05:
                return t_ball, pt

        t_me = arrival_time(player.position, vel, pt, reach, vmax, accel)
        return t_me, pt

    def select_chaser(self, obs):
        cands = [p for p in obs.my_players if p.id != 0]
        self._icpt = {p.id: self.player_intercept(obs, p) for p in cands}
        best = min(cands, key=lambda p: self._icpt[p.id][0])

        if self._chaser_id is not None:
            prev = next((p for p in cands if p.id == self._chaser_id), None)
            if prev is not None and prev.id != best.id:
                if self._icpt[prev.id][0] <= self._icpt[best.id][0] + self.CHASER_HYSTERESIS:
                    best = prev
        self._chaser_id = best.id
        return best

    @staticmethod
    def _carrier(obs):
        holder = obs.ball.controlling_player
        if holder is None or holder < len(obs.my_players):
            return None
        return next((o for o in obs.opponents if o.id == holder), None)

    def chase_target(self, obs, player):
        """Where the chaser runs: the intercept point, or (jockey) the goal side of a carrier."""
        tgt = self._icpt.get(player.id, (0.0, obs.ball.position))[1]
        carrier = self._carrier(obs)
        if carrier is None or obs.ball.position[0] > self.JOCKEY_BALL_X:
            return tgt
        f = obs.field
        my_goal = f.my_goal
        if distance(player.position, my_goal) > distance(carrier.position, my_goal) + 1.0:
            return tgt
        gap = f.kick_range * self.JOCKEY_GAP
        if distance(player.position, carrier.position) <= gap + 1.0:
            return tgt
        ug = _unit(carrier.position, my_goal)
        return (carrier.position[0] + ug[0] * gap, carrier.position[1] + ug[1] * gap)

    def _attacking(self, obs, n):
        ball = obs.ball
        team = ball.controlling_team
        holder = ball.controlling_player
        raw = (team == 0)
        mine_has = holder is not None and holder < n
        raw = raw and (mine_has or closest_to_ball(obs))

        # Dwell: don't flip role every tick, except when possession truly changes.
        if team != self._last_team:
            self._mode_attacking = raw
            self._mode_tick = obs.tick
        elif raw != self._mode_attacking and obs.tick - self._mode_tick >= self.MODE_DWELL_TICKS:
            self._mode_attacking = raw
            self._mode_tick = obs.tick
        self._last_team = team
        return self._mode_attacking

    def act(self, obs):
        n = len(obs.my_players)
        if obs.tick < self._last_tick:
            self._reset_state()
        self._last_tick = obs.tick

        for ev in obs.events:
            if ev.get("kind") == "goal":
                self._we_scored = (ev.get("team") == 0)
                self._seen_centre = False
                self._chaser_id = None

        ball = obs.ball
        bx, by = ball.position
        centre_dead = (abs(bx) < 1.0 and abs(by) < 1.0
                       and math.hypot(ball.velocity[0], ball.velocity[1]) < 0.5)
        if self._we_scored:
            if centre_dead:
                self._seen_centre = True
            elif self._seen_centre:
                self._we_scored = False
                self._seen_centre = False

        # Their kickoff: hold shape, never enter the circle
        if centre_dead and (ball.controlling_team != 0 or self._we_scored):
            return self._hold_for_kickoff(obs)

        self._path = predict_ball_path(obs)
        chaser = self.select_chaser(obs)

        attacking = self._attacking(obs, n)
        if attacking:
            return self.on_the_ball(obs)
        return self.off_the_ball(obs)

    def _hold_for_kickoff(self, obs):
        f = obs.field
        actions = TeamAction()
        r = f.centre_circle_radius + 1.5
        for p in obs.my_players:
            if p.id == 0:
                actions.set(p.id, PlayerAction(
                    movement=self._go(p.position, (f.my_goal[0] + 2.5, 0.0))))
                continue
            d = math.hypot(p.position[0], p.position[1])
            if d < r:
                if d < 1e-6:
                    mv = (-1.0, 0.0)
                else:
                    mv = (p.position[0] / d, p.position[1] / d)
                actions.set(p.id, PlayerAction(movement=mv))
            else:
                actions.set(p.id, PlayerAction())
        return actions

    def keeper_action(self, obs, player):
        f = obs.field
        ball_pos = obs.ball.position
        my_goal = f.my_goal
        mouth = f.goal_width / 2.0
        safe_mouth = mouth - 1.2

        # 1. Ball at keeper's feet: distribute forward or clear wide
        if obs.can_kick(player.id):
            best_pass = self.best_pass_option(obs, player, min_fwd=10.0)
            if best_pass is not None:
                mate, target_pos, power = best_pass
                return PlayerAction(
                    movement=_unit(player.position, target_pos),
                    kick_direction=_unit(player.position, target_pos),
                    kick_power=power,
                )
            clear_y = 20.0 if player.position[1] >= 0.0 else -20.0
            clear_target = (15.0, clear_y)
            return PlayerAction(
                movement=_unit(player.position, clear_target),
                kick_direction=_unit(player.position, clear_target),
                kick_power=0.95,
            )

        # 2. Shot/ball rolling towards goal mouth: step on its line
        line_x = my_goal[0] + 1.2
        for px, py in self._path:
            if px <= line_x:
                if abs(py) <= mouth + 0.5:
                    return PlayerAction(movement=self._go(
                        player.position, (px, clamp(py, -safe_mouth, safe_mouth)), 0.5))
                break

        # 3. Loose ball in the box and nearest: claim it
        d_ball = distance(player.position, ball_pos)
        holder = obs.ball.controlling_player
        in_box = ball_pos[0] < my_goal[0] + 16.0 and abs(ball_pos[1]) < mouth + 4.0
        if in_box and d_ball < 9.0 and (holder is None or d_ball < 5.0):
            if obs.closest_my_player_to(ball_pos).id == player.id:
                return PlayerAction(movement=self._go(player.position, ball_pos, 0.5))

        # 4. Cut the angle: stand on line between ball and goal center
        ux, uy = ball_pos[0] - my_goal[0], ball_pos[1] - my_goal[1]
        d = math.hypot(ux, uy)
        if d > 1e-6:
            r = clamp(0.22 * d, 2.0, 6.5)
            spot = (max(my_goal[0] + 1.5, my_goal[0] + ux / d * r),
                    clamp(my_goal[1] + uy / d * r, -safe_mouth, safe_mouth))
            return PlayerAction(movement=self._go(player.position, spot, 2.0))

        stand_x = clamp(my_goal[0] + 2.5 + 1.5 * (1.0 - min(1.0, distance(my_goal, ball_pos) / 50.0)),
                        my_goal[0] + 2.0, my_goal[0] + 4.5)
        stand_y = clamp(ball_pos[1] * 0.42, -safe_mouth, safe_mouth)
        return PlayerAction(movement=self._go(player.position, (stand_x, stand_y), 2.0))

    def lead_target(self, obs, passer, mate):
        hz = obs.field.simulation_hz
        hx = obs.field.width / 2.0 - 1.0
        hy = obs.field.height / 2.0 - 1.0

        target = mate.position
        dist = distance(passer.position, target)
        power = clamp(dist / 26.0, 0.55, 0.95)

        for _ in range(2):
            t = obs.ticks_to_cover(dist, power) / hz
            target = (
                clamp(mate.position[0] + mate.velocity[0] * t, -hx, hx),
                clamp(mate.position[1] + mate.velocity[1] * t, -hy, hy),
            )
            dist = distance(passer.position, target)
            power = clamp(dist / 26.0, 0.55, 0.95)

        return target, power

    def is_pass_safe(self, obs, from_pos, to_pos, power, receiver):
        f = obs.field
        total_dist = distance(from_pos, to_pos)
        if total_dist < 3.0:
            return False

        t_dest = obs.ticks_to_cover(total_dist, power) / f.simulation_hz

        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        for frac in (0.25, 0.5, 0.75):
            d_frac = total_dist * frac
            t_ball = obs.ticks_to_cover(d_frac, power) / f.simulation_hz
            pt = (from_pos[0] + dx * frac, from_pos[1] + dy * frac)
            for opp in obs.opponents:
                d_opp = max(0.0, distance(opp.position, pt) - f.kick_range)
                t_opp = d_opp / (f.max_speed * 0.75) + 0.12
                if t_opp <= t_ball:
                    return False

        d_rcv = max(0.0, distance(receiver.position, to_pos) - f.kick_range)
        t_rcv = d_rcv / f.max_speed
        for opp in obs.opponents:
            d_opp = max(0.0, distance(opp.position, to_pos) - f.kick_range)
            t_opp = d_opp / (f.max_speed * 0.8)
            if t_opp < t_dest - 0.1 and t_opp < t_rcv:
                return False

        return True

    def best_pass_option(self, obs, player, min_fwd=0.0):
        my_goal = obs.field.my_goal
        best_candidate = None
        best_score = -1e9

        for mate in obs.my_players:
            if mate.id == player.id or mate.id == 0:
                continue

            tpos, pwr = self.lead_target(obs, player, mate)
            fwd_gain = tpos[0] - player.position[0]
            if fwd_gain < min_fwd:
                continue

            if tpos[0] < my_goal[0] + 14.0 and abs(tpos[1]) < obs.field.goal_width / 2.0 + 2.0:
                continue

            if not self.is_pass_safe(obs, player.position, tpos, pwr, mate):
                continue

            space = min((distance(tpos, o.position) for o in obs.opponents), default=20.0)
            score = (fwd_gain * 1.5) + (space * 1.0) - (distance(player.position, tpos) * 0.08)
            if tpos[0] > 10.0:
                score += 5.0

            if score > best_score:
                best_score = score
                best_candidate = (mate, tpos, pwr)

        return best_candidate

    def get_best_shot(self, obs, player):
        """Best open corner with keeper reach time vs. ball flight time gating."""
        f = obs.field
        opp_goal = f.opponent_goal
        gx = opp_goal[0]
        gw = f.goal_width / 2.0 - 0.9

        targets = [(gx, gw), (gx, -gw), (gx, gw * 0.5), (gx, -gw * 0.5), (gx, 0.0)]

        opp_keeper = min(obs.opponents, key=lambda o: distance(o.position, opp_goal))
        best_target = None
        best_score = -1e9
        self._shot_margin = -1e9

        for t in targets:
            blocked = False
            for opp in obs.opponents:
                if opp.id == opp_keeper.id:
                    continue
                d_seg = closest_point_on_segment(opp.position, player.position, t)
                if distance(opp.position, d_seg) < f.player_radius * 1.15:
                    blocked = True
                    break
            if blocked:
                continue

            dist_shot = distance(player.position, t)
            shot_time = dist_shot / (f.kick_impulse * 0.95)

            shot_dx = t[0] - player.position[0]
            if shot_dx > 1e-4:
                frac_x = clamp((opp_keeper.position[0] - player.position[0]) / shot_dx, 0.0, 1.0)
                pass_y = player.position[1] + frac_x * (t[1] - player.position[1])
                keeper_reach_dist = max(0.0, abs(opp_keeper.position[1] - pass_y) - f.kick_range)
                keeper_time = keeper_reach_dist / (f.max_speed * 0.8) + 0.15
            else:
                keeper_reach_dist = 0.0
                keeper_time = 0.0

            margin = keeper_time - shot_time
            score = (keeper_reach_dist * 3.0) + (margin * 4.0) + (abs(t[1]) * 1.0)
            if score > best_score:
                best_score = score
                best_target = t
                self._shot_margin = margin

        return best_target

    def find_box_pass(self, obs, player):
        opp_goal = obs.field.opponent_goal
        for mate in obs.my_players:
            if mate.id == 0 or mate.id == player.id:
                continue
            if mate.position[0] < player.position[0] - 2.0:
                continue
            if distance(mate.position, opp_goal) > 28.0:
                continue
            tpos, pwr = self.lead_target(obs, player, mate)
            if self.is_pass_safe(obs, player.position, tpos, pwr, mate):
                saved = self._shot_margin
                shot = self.get_best_shot(obs, mate)
                self._shot_margin = saved
                if shot is not None:
                    return mate, tpos, pwr
        return None

    def find_dribble_vector(self, obs, player):
        gx, gy = obs.field.opponent_goal
        dx, dy = gx - player.position[0], gy - player.position[1]
        length = math.hypot(dx, dy)
        if length > 0.0:
            dx, dy = dx / length, dy / length

        for opp in obs.opponents:
            dist = distance(player.position, opp.position)
            if 0.1 < dist < 9.0:
                rx = player.position[0] - opp.position[0]
                ry = player.position[1] - opp.position[1]
                rlen = math.hypot(rx, ry)
                if rlen < 1e-9:
                    continue
                weight = 2.5 / (dist + 0.2)
                dx += (rx / rlen) * weight
                dy += (ry / rlen) * weight

        d_final = math.hypot(dx, dy)
        if d_final > 0.0:
            return (dx / d_final, dy / d_final)
        return (1.0, 0.0)

    def pick_lane_guard(self, obs, chaser_id):
        """One non-chaser outfield player blocks the line from ball to our goal."""
        f = obs.field
        ball = obs.ball.position
        if ball[0] > self.JOCKEY_BALL_X:
            return None, None
        gx, gy = f.my_goal
        d = distance(ball, (gx, gy))
        if d < self.GUARD_DIST + 3.0:
            return None, None
        spot = (ball[0] + (gx - ball[0]) / d * self.GUARD_DIST,
                ball[1] + (gy - ball[1]) / d * self.GUARD_DIST)
        cands = [p for p in obs.my_players if p.id != 0 and p.id != chaser_id]
        if len(cands) < 2:
            return None, None
        best = min(cands, key=lambda p: distance(p.position, spot))
        prev = next((p for p in cands if p.id == self._guard_prev), None)
        if (prev is not None and prev.id != best.id
                and distance(prev.position, spot) <= distance(best.position, spot) + self.GUARD_STICKY):
            best = prev
        self._guard_prev = best.id
        return best.id, spot

    def assign_marks(self, obs, markers):
        """Return {player_id: spot} for every marker using sticky permutation matching."""
        f = obs.field
        my_goal = f.my_goal
        bx, by = obs.ball.position
        hx, hy = f.width / 2.0, f.height / 2.0
        spots = {}
        if not markers:
            return spots

        if not obs.opponents:
            return self._cover_spots(obs, markers, {})

        keeper_id = min(o.id for o in obs.opponents)
        carrier_id = obs.ball.controlling_player
        if carrier_id is None or carrier_id < len(obs.my_players):
            near = min(obs.opponents, key=lambda o: distance(o.position, obs.ball.position))
            carrier_id = near.id if distance(near.position, obs.ball.position) < self.CARRIER_RADIUS else None

        cands = [o for o in obs.opponents
                 if o.id != keeper_id and o.id != carrier_id and o.position[0] < self.MARK_MAX_X]
        cands.sort(key=lambda o: distance(o.position, my_goal))
        targets = cands[:len(markers)]

        pairs = {}
        if targets:
            best_cost, best_pairs = 1e18, {}
            for chosen in permutations(markers, len(targets)):
                cost = sum(distance(m.position, t.position) for m, t in zip(chosen, targets))
                if cost < best_cost:
                    best_cost = cost
                    best_pairs = {m.id: t for m, t in zip(chosen, targets)}
            pairs = best_pairs

            # Sticky matching to prevent flickering swaps
            prev = self._marks_prev
            if prev and len(prev) == len(pairs):
                by_id = {t.id: t for t in targets}
                keep, prev_cost, ok = {}, 0.0, True
                m_by_id = {m.id: m for m in markers}
                for pid, oid in prev.items():
                    if pid not in m_by_id or oid not in by_id:
                        ok = False
                        break
                    keep[pid] = by_id[oid]
                    prev_cost += distance(m_by_id[pid].position, by_id[oid].position)
                if ok and len({t.id for t in keep.values()}) == len(keep) and prev_cost <= best_cost + self.MARK_STICKY:
                    pairs = keep
        self._marks_prev = {pid: t.id for pid, t in pairs.items()}

        for pid, opp in pairs.items():
            ug = _unit(opp.position, my_goal)
            ub = _unit(opp.position, obs.ball.position)
            vx, vy = 0.65 * ug[0] + 0.35 * ub[0], 0.65 * ug[1] + 0.35 * ub[1]
            ln = math.hypot(vx, vy)
            if ln < 1e-6:
                vx, vy = ug
            else:
                vx, vy = vx / ln, vy / ln
            spots[pid] = (
                clamp(opp.position[0] + vx * 2.4, my_goal[0] + 3.0, hx - 5.0),
                clamp(opp.position[1] + vy * 2.4, -hy + 2.0, hy - 2.0),
            )

        spare = [m for m in markers if m.id not in spots]
        spots.update(self._cover_spots(obs, spare, spots))
        return spots

    def _cover_spots(self, obs, spare, taken):
        out = {}
        if not spare:
            return out
        f = obs.field
        bx, by = obs.ball.position
        hy = f.height / 2.0
        x = clamp(bx - 9.0, f.my_goal[0] + 10.0, 6.0)
        offsets = {1: [0.0], 2: [-9.0, 9.0], 3: [-12.0, 0.0, 12.0]}.get(len(spare))
        if offsets is None:
            offsets = [(-12.0 + 24.0 * i / max(1, len(spare) - 1)) for i in range(len(spare))]
        for m, off in zip(sorted(spare, key=lambda p: p.position[1]), offsets):
            out[m.id] = (x, clamp(by * 0.4 + off, -hy + 4.0, hy - 4.0))
        return out

    def support_base(self, obs, player):
        f = obs.field
        bx, by = obs.ball.position
        gx = f.my_goal[0]
        hx, hy = f.width / 2.0, f.height / 2.0
        pid = player.id
        if pid == 1:
            return (clamp(bx - 13.0, gx + 8.0, 2.0), clamp(by * 0.35, -10.0, 10.0))
        if pid == 2:
            return (clamp(bx + 4.0, -18.0, hx - 8.0),
                    clamp(by * 0.3 - 15.0, -hy + 3.0, -4.0))
        if pid == 3:
            return (clamp(bx + 4.0, -18.0, hx - 8.0),
                    clamp(by * 0.3 + 15.0, 4.0, hy - 3.0))
        return (clamp(bx + 16.0, -5.0, hx - 4.0), clamp(-by * 0.45, -12.0, 12.0))

    def support_targets(self, obs, supporters):
        f = obs.field
        hx, hy = f.width / 2.0, f.height / 2.0
        bx, by = obs.ball.position
        out = {}
        chosen = []
        for p in sorted(supporters, key=lambda q: q.id):
            base = self.support_base(obs, p)
            if p.id == 1:
                out[p.id] = base
                chosen.append(base)
                continue
            prev = self._sup_prev.get(p.id)
            best, best_s = base, -1e9
            for ox in (0.0, 4.0, 8.0):
                for oy in (-6.0, 0.0, 6.0):
                    spot = (clamp(base[0] + ox, -hx + 3.0, hx - 4.0),
                            clamp(base[1] + oy, -hy + 3.0, hy - 3.0))
                    s = self._score_support(obs, spot, (ox, oy), chosen, prev, bx)
                    if s > best_s:
                        best_s, best = s, spot
            self._sup_prev[p.id] = best
            out[p.id] = best
            chosen.append(best)
        return out

    def _score_support(self, obs, spot, off, chosen, prev, bx):
        ball = obs.ball.position
        opp_d, lane = 12.0, 6.0
        for o in obs.opponents:
            d = distance(spot, o.position)
            if d < opp_d:
                opp_d = d
            cp = closest_point_on_segment(o.position, ball, spot)
            ld = distance(o.position, cp)
            if ld < lane:
                lane = ld
        s = opp_d
        s += lane * 0.6
        if lane < 2.0:
            s -= 5.0
        s += clamp(spot[0] - bx, -10.0, 20.0) * 0.08
        s -= 0.25 * math.hypot(off[0], off[1])
        for c in chosen:
            dd = distance(spot, c)
            if dd < 7.0:
                s -= (7.0 - dd) * 0.7
        if prev is not None and distance(prev, spot) < 2.5:
            s += 1.2
        return s

    def chaser_action(self, obs, player):
        f = obs.field
        if not obs.can_kick(player.id):
            tgt = self.chase_target(obs, player)
            return PlayerAction(movement=self._go(player.position, tgt, 0.8))

        opp_goal = f.opponent_goal
        goal_dist = distance(player.position, opp_goal)

        carrier = self._carrier(obs)
        opp_holds = carrier is not None
        pressed = opp_holds or any(
            distance(player.position, o.position) < 7.0 for o in obs.opponents)

        # Deep in our half under pressure: clear wide
        if player.position[0] < -20.0 and pressed and obs.ball.controlling_team == 1:
            clear_y = 20.0 if player.position[1] >= 0.0 else -20.0
            tgt = (15.0, clear_y)
            return PlayerAction(
                movement=_unit(player.position, tgt),
                kick_direction=_unit(player.position, tgt),
                kick_power=0.9,
            )

        # In opponent possession: tackle & shoot/clear
        if obs.ball.controlling_team == 1:
            shot = self.get_best_shot(obs, player)
            if shot is not None:
                return PlayerAction(kick_direction=_unit(player.position, shot), kick_power=1.0)
            pass_opt = self.best_pass_option(obs, player, min_fwd=0.0)
            if pass_opt is not None:
                mate, tpos, pwr = pass_opt
                return PlayerAction(kick_direction=_unit(player.position, tpos), kick_power=pwr)
            return PlayerAction(kick_direction=_unit(player.position, opp_goal), kick_power=0.85)

        # Ball is ours: shoot if sight of goal is open and not gated
        if goal_dist <= self.SHOOT_RANGE:
            shot = self.get_best_shot(obs, player)
            if shot is not None:
                gate_ok = (goal_dist <= self.SHOT_GATE_RANGE or self._shot_margin > -0.05)
                if gate_ok:
                    return PlayerAction(kick_direction=_unit(player.position, shot), kick_power=1.0)
            alt = self.find_box_pass(obs, player)
            if alt is not None:
                mate, tpos, pwr = alt
                return PlayerAction(kick_direction=_unit(player.position, tpos), kick_power=pwr)

        # Progressive pass
        pass_info = self.best_pass_option(obs, player, min_fwd=4.0)
        if pass_info is not None:
            mate, tpos, pwr = pass_info
            return PlayerAction(kick_direction=_unit(player.position, tpos), kick_power=pwr)

        # Steered obstacle-repulsion dribble
        steer = self.find_dribble_vector(obs, player)
        return PlayerAction(movement=steer, kick_direction=steer, kick_power=0.38)

    def on_the_ball(self, obs):
        actions = TeamAction()
        chaser_id = self._chaser_id or self.select_chaser(obs).id
        outfield = [p for p in obs.my_players if p.id != 0]
        supporters = [p for p in outfield if p.id != chaser_id]
        targets = self.support_targets(obs, supporters)

        for player in obs.my_players:
            if player.id == 0:
                actions.set(player.id, self.keeper_action(obs, player))
            elif player.id == chaser_id:
                actions.set(player.id, self.chaser_action(obs, player))
            else:
                actions.set(player.id, PlayerAction(movement=self._go(player.position, targets[player.id])))

        return actions

    def off_the_ball(self, obs):
        actions = TeamAction()
        chaser_id = self._chaser_id or self.select_chaser(obs).id
        outfield = [p for p in obs.my_players if p.id != 0]

        guard_id, guard_spot = self.pick_lane_guard(obs, chaser_id)
        markers = [p for p in outfield if p.id != chaser_id and p.id != guard_id]
        spots = self.assign_marks(obs, markers)

        for player in obs.my_players:
            if player.id == 0:
                actions.set(player.id, self.keeper_action(obs, player))
            elif player.id == chaser_id:
                tgt = self.chase_target(obs, player)
                actions.set(player.id, PlayerAction(movement=self._go(player.position, tgt, 0.8)))
            elif player.id == guard_id:
                actions.set(player.id, PlayerAction(movement=self._go(player.position, guard_spot, 2.0)))
            else:
                target_spot = spots.get(player.id, self.support_base(obs, player))
                actions.set(player.id, PlayerAction(movement=self._go(player.position, target_spot, 2.0)))

        return actions