# cython: boundscheck=False, wraparound=False, cdivision=True
"""Vectorized Slither.io game engine in Cython for high-performance RL."""

import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, sqrt, atan2, M_PI

cnp.import_array()

DEF MAX_SEG = 200
DEF MAX_FOOD = 1024
DEF NUM_NPCS = 4
DEF K_FOOD = 16
DEF K_NPC = 8
DEF OBS_DIM_VAL = 54  # 6 + K_FOOD*2 + K_NPC*2


# ---------- RNG helpers ----------

cdef inline unsigned long long _xorshift64(unsigned long long* state) noexcept nogil:
    cdef unsigned long long x = state[0]
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    state[0] = x
    return x

cdef inline double _rand_double(unsigned long long* state) noexcept nogil:
    cdef unsigned long long mask = 0x1FFFFFFFFFFFFF
    cdef unsigned long long divisor = 0x20000000000000
    return <double>(_xorshift64(state) & mask) / <double>divisor


# ---------- Main engine ----------

cdef class VecSlither:
    """Vectorized Slither game engine managing N parallel environments."""

    cdef readonly int n_envs, obs_dim

    # Game config
    cdef double arena_radius, base_speed, turn_rate
    cdef double head_radius, body_radius, food_radius
    cdef double boost_mass_cost, segment_spacing, viewport
    cdef int initial_length, max_steps, initial_food, food_respawn_rate
    cdef double death_food_frac
    cdef int npc_respawn_delay

    # Reward config (unused — reward is pure delta-length, kept for API compat)
    cdef double r_food, r_kill, r_death_scale, r_survival, r_boost_cost

    # Player state  (n_envs,)
    cdef double[::1] px, py, pdir, pscore, pboost_debt
    cdef int[::1] plength, palive, pboosting, pstep
    cdef double[:,::1] pseg_x, pseg_y          # (n_envs, MAX_SEG)
    cdef int[::1] pseg_head

    # NPC state  (n_envs * NUM_NPCS,)
    cdef double[::1] nx, ny, ndir, nboost_debt
    cdef int[::1] nlength, nalive, nrespawn
    cdef double[:,::1] nseg_x, nseg_y          # (n_total_npcs, MAX_SEG)
    cdef int[::1] nseg_head

    # Food  (n_envs, MAX_FOOD)
    cdef double[:,::1] fx, fy
    cdef int[:,::1] factive

    # Output buffers
    cdef float[:,::1] obs_buf
    cdef float[::1] rew_buf
    cdef cnp.uint8_t[::1] done_buf

    # Episode tracking
    cdef double[::1] ep_return
    cdef int[::1] ep_len
    cdef float[::1] ep_ret_buf
    cdef int[::1] ep_len_buf, ep_slen_buf

    # Per-env RNG
    cdef unsigned long long[::1] rng_state

    # ------------------------------------------------------------------ init
    def __init__(self, int n_envs, int seed=42,
                 double food_reward=1.5, double kill_reward=10.0,
                 double death_scale=0.1, double survival_bonus=-0.005,
                 double boost_cost=-0.01):
        self.n_envs = n_envs
        self.obs_dim = OBS_DIM_VAL

        # Config
        self.arena_radius = 1000.0
        self.base_speed   = 3.0
        self.turn_rate    = 0.12
        self.head_radius  = 8.0
        self.body_radius  = 6.5
        self.food_radius  = 8.0
        self.boost_mass_cost = 0.2
        self.segment_spacing = 4.0
        self.initial_length  = 10
        self.max_steps       = 4000
        self.initial_food    = 800
        self.food_respawn_rate = 3
        self.death_food_frac   = 0.8
        self.npc_respawn_delay = 30
        self.viewport = 200.0

        # Rewards
        self.r_food        = food_reward
        self.r_kill        = kill_reward
        self.r_death_scale = death_scale
        self.r_survival    = survival_bonus
        self.r_boost_cost  = boost_cost

        # ---------- allocate arrays ----------
        cdef int n_total = n_envs * NUM_NPCS

        self.px  = np.zeros(n_envs, dtype=np.float64)
        self.py  = np.zeros(n_envs, dtype=np.float64)
        self.pdir = np.zeros(n_envs, dtype=np.float64)
        self.pscore = np.zeros(n_envs, dtype=np.float64)
        self.pboost_debt = np.zeros(n_envs, dtype=np.float64)
        self.plength   = np.zeros(n_envs, dtype=np.intc)
        self.palive    = np.ones(n_envs, dtype=np.intc)
        self.pboosting = np.zeros(n_envs, dtype=np.intc)
        self.pstep     = np.zeros(n_envs, dtype=np.intc)
        self.pseg_x    = np.zeros((n_envs, MAX_SEG), dtype=np.float64)
        self.pseg_y    = np.zeros((n_envs, MAX_SEG), dtype=np.float64)
        self.pseg_head = np.zeros(n_envs, dtype=np.intc)

        self.nx  = np.zeros(n_total, dtype=np.float64)
        self.ny  = np.zeros(n_total, dtype=np.float64)
        self.ndir = np.zeros(n_total, dtype=np.float64)
        self.nboost_debt = np.zeros(n_total, dtype=np.float64)
        self.nlength  = np.zeros(n_total, dtype=np.intc)
        self.nalive   = np.zeros(n_total, dtype=np.intc)
        self.nrespawn = np.zeros(n_total, dtype=np.intc)
        self.nseg_x   = np.zeros((n_total, MAX_SEG), dtype=np.float64)
        self.nseg_y   = np.zeros((n_total, MAX_SEG), dtype=np.float64)
        self.nseg_head = np.zeros(n_total, dtype=np.intc)

        self.fx = np.zeros((n_envs, MAX_FOOD), dtype=np.float64)
        self.fy = np.zeros((n_envs, MAX_FOOD), dtype=np.float64)
        self.factive = np.zeros((n_envs, MAX_FOOD), dtype=np.intc)

        self.obs_buf  = np.zeros((n_envs, OBS_DIM_VAL), dtype=np.float32)
        self.rew_buf  = np.zeros(n_envs, dtype=np.float32)
        self.done_buf = np.zeros(n_envs, dtype=np.uint8)

        self.ep_return   = np.zeros(n_envs, dtype=np.float64)
        self.ep_len      = np.zeros(n_envs, dtype=np.intc)
        self.ep_ret_buf  = np.zeros(n_envs, dtype=np.float32)
        self.ep_len_buf  = np.zeros(n_envs, dtype=np.intc)
        self.ep_slen_buf = np.zeros(n_envs, dtype=np.intc)

        # Per-env RNG seeds (must be non-zero)
        rng = np.zeros(n_envs, dtype=np.uint64)
        cdef int i
        for i in range(n_envs):
            rng[i] = <unsigned long long>(<unsigned long long>seed * 6364136223846793005ULL
                                           + <unsigned long long>i * 1442695040888963407ULL + 1ULL)
            if rng[i] == 0:
                rng[i] = 1
        self.rng_state = rng

        self.reset_all()

    # -------------------------------------------------------- public reset
    def reset_all(self):
        """Reset every environment.  Returns initial obs (n_envs, obs_dim)."""
        cdef int e
        for e in range(self.n_envs):
            self._reset_env(e)
            self._compute_obs_env(e)
        return np.asarray(self.obs_buf).copy()

    # -------------------------------------------------------- public step
    def step(self, cnp.ndarray[int, ndim=1] actions):
        """Step all envs.  Returns (obs, rewards, dones, ep_returns, ep_lens, ep_snake_lens)."""
        cdef int[::1] act = actions
        cdef int e
        for e in range(self.n_envs):
            self.done_buf[e] = 0
            self.rew_buf[e] = 0.0
            self.ep_ret_buf[e] = 0.0
            self.ep_len_buf[e] = 0
            self.ep_slen_buf[e] = 0
            self._step_env(e, act[e])
            self._compute_obs_env(e)
        return (np.asarray(self.obs_buf).copy(),
                np.asarray(self.rew_buf).copy(),
                np.asarray(self.done_buf).copy(),
                np.asarray(self.ep_ret_buf).copy(),
                np.asarray(self.ep_len_buf).copy(),
                np.asarray(self.ep_slen_buf).copy())

    # ============================================================ internals
    # -------------------------------------------------------- reset one env
    cdef void _reset_env(self, int e) noexcept nogil:
        cdef unsigned long long* rng = &self.rng_state[e]
        cdef double angle, radius, d
        cdef int k

        # Player spawn
        angle  = _rand_double(rng) * 2.0 * M_PI
        radius = sqrt(_rand_double(rng)) * self.arena_radius * 0.7
        self.px[e]  = cos(angle) * radius
        self.py[e]  = sin(angle) * radius
        self.pdir[e] = _rand_double(rng) * 2.0 * M_PI
        self.plength[e]  = self.initial_length
        self.palive[e]   = 1
        self.pboosting[e] = 0
        self.pstep[e]    = 0
        self.pscore[e]   = 0.0
        self.pboost_debt[e] = 0.0
        self.pseg_head[e] = self.initial_length - 1

        d = self.pdir[e]
        for k in range(self.initial_length):
            self.pseg_x[e, k] = self.px[e] + <double>(k - self.initial_length + 1) * self.segment_spacing * cos(d)
            self.pseg_y[e, k] = self.py[e] + <double>(k - self.initial_length + 1) * self.segment_spacing * sin(d)

        # NPCs
        for k in range(NUM_NPCS):
            self._spawn_npc(e, k)

        # Food: clear then spawn initial_food items
        cdef int fi
        for fi in range(MAX_FOOD):
            self.factive[e, fi] = 0
        for fi in range(self.initial_food):
            if fi >= MAX_FOOD:
                break
            self._spawn_food_slot(e, fi)
            self.factive[e, fi] = 1

        self.ep_return[e] = 0.0
        self.ep_len[e]    = 0

    # -------------------------------------------------------- spawn helpers
    cdef void _spawn_npc(self, int e, int npc_id) noexcept nogil:
        cdef int idx = e * NUM_NPCS + npc_id
        cdef unsigned long long* rng = &self.rng_state[e]
        cdef double angle, radius, d
        cdef int k

        angle  = _rand_double(rng) * 2.0 * M_PI
        radius = sqrt(_rand_double(rng)) * self.arena_radius * 0.7
        self.nx[idx]  = cos(angle) * radius
        self.ny[idx]  = sin(angle) * radius
        self.ndir[idx] = _rand_double(rng) * 2.0 * M_PI
        self.nlength[idx]  = self.initial_length
        self.nalive[idx]   = 1
        self.nrespawn[idx] = 0
        self.nboost_debt[idx] = 0.0
        self.nseg_head[idx] = self.initial_length - 1

        d = self.ndir[idx]
        for k in range(self.initial_length):
            self.nseg_x[idx, k] = self.nx[idx] + <double>(k - self.initial_length + 1) * self.segment_spacing * cos(d)
            self.nseg_y[idx, k] = self.ny[idx] + <double>(k - self.initial_length + 1) * self.segment_spacing * sin(d)

    cdef void _spawn_food_slot(self, int e, int slot) noexcept nogil:
        cdef unsigned long long* rng = &self.rng_state[e]
        cdef double angle = _rand_double(rng) * 2.0 * M_PI
        cdef double radius = sqrt(_rand_double(rng)) * self.arena_radius * 0.95
        self.fx[e, slot] = cos(angle) * radius
        self.fy[e, slot] = sin(angle) * radius

    # -------------------------------------------------------- full step
    cdef void _step_env(self, int e, int action) noexcept nogil:
        cdef unsigned long long* rng = &self.rng_state[e]
        cdef int turn_action, wants_boost
        cdef double reward = 0.0
        cdef int food_eaten = 0, kills = 0, died = 0
        cdef double dx, dy, dist_sq, eat_r_sq
        cdef int fi, ni, nidx, k, seg_idx, new_head
        cdef int length_before = self.plength[e]

        self.pstep[e] += 1

        # ---- parse action ----
        turn_action = action % 3
        wants_boost = 1 if action >= 3 else 0

        # ---- turn ----
        if turn_action == 1:
            self.pdir[e] -= self.turn_rate
        elif turn_action == 2:
            self.pdir[e] += self.turn_rate
        # normalise
        if self.pdir[e] > M_PI:
            self.pdir[e] -= 2.0 * M_PI
        if self.pdir[e] < -M_PI:
            self.pdir[e] += 2.0 * M_PI

        # ---- boost ----
        if wants_boost and self.plength[e] > self.initial_length:
            self.pboosting[e] = 1
        else:
            self.pboosting[e] = 0

        # ---- move player ----
        if self.pboosting[e]:
            # Two sub-steps at base speed (same as original)
            self.px[e] += cos(self.pdir[e]) * self.base_speed
            self.py[e] += sin(self.pdir[e]) * self.base_speed
            new_head = (self.pseg_head[e] + 1) % MAX_SEG
            self.pseg_head[e] = new_head
            self.pseg_x[e, new_head] = self.px[e]
            self.pseg_y[e, new_head] = self.py[e]

            self.px[e] += cos(self.pdir[e]) * self.base_speed
            self.py[e] += sin(self.pdir[e]) * self.base_speed
            new_head = (self.pseg_head[e] + 1) % MAX_SEG
            self.pseg_head[e] = new_head
            self.pseg_x[e, new_head] = self.px[e]
            self.pseg_y[e, new_head] = self.py[e]

            # Boost mass cost
            self.pboost_debt[e] += self.boost_mass_cost
            if self.pboost_debt[e] >= 1.0 and self.plength[e] > 3:
                self.plength[e] -= 1
                self.pscore[e] -= 1.0
                if self.pscore[e] < 0.0:
                    self.pscore[e] = 0.0
                self.pboost_debt[e] -= 1.0
        else:
            self.px[e] += cos(self.pdir[e]) * self.base_speed
            self.py[e] += sin(self.pdir[e]) * self.base_speed
            new_head = (self.pseg_head[e] + 1) % MAX_SEG
            self.pseg_head[e] = new_head
            self.pseg_x[e, new_head] = self.px[e]
            self.pseg_y[e, new_head] = self.py[e]

        # ---- NPC AI + move ----
        self._step_npcs(e)

        # ---- player eats food ----
        eat_r_sq = (self.head_radius + self.food_radius) * (self.head_radius + self.food_radius)
        for fi in range(MAX_FOOD):
            if not self.factive[e, fi]:
                continue
            dx = self.fx[e, fi] - self.px[e]
            dy = self.fy[e, fi] - self.py[e]
            dist_sq = dx * dx + dy * dy
            if dist_sq < eat_r_sq:
                self.factive[e, fi] = 0
                food_eaten += 1
                if self.plength[e] < MAX_SEG:
                    self.plength[e] += 1
                self.pscore[e] += 1.0
        # (food_eaten already reflected in plength above)

        # ---- NPCs eat food ----
        self._npc_eat_food(e)

        # ---- collision: player vs boundary ----
        dist_sq = self.px[e] * self.px[e] + self.py[e] * self.py[e]
        if dist_sq > self.arena_radius * self.arena_radius:
            died = 1

        # ---- collision: player head vs NPC body ----
        if not died:
            died = self._player_hits_npc_body(e)

        # ---- collision: NPC head vs player body -> kills ----
        if not died:
            kills = self._npc_hits_player_body(e)

        # ---- NPC boundary / cleanup ----
        self._npc_boundary_check(e)

        # ---- food respawn ----
        cdef int spawned = 0
        for fi in range(MAX_FOOD):
            if spawned >= self.food_respawn_rate:
                break
            if not self.factive[e, fi]:
                self._spawn_food_slot(e, fi)
                self.factive[e, fi] = 1
                spawned += 1

        # ---- NPC respawn ----
        cdef int npc_base = e * NUM_NPCS
        for ni in range(NUM_NPCS):
            nidx = npc_base + ni
            if not self.nalive[nidx]:
                self.nrespawn[nidx] -= 1
                if self.nrespawn[nidx] <= 0:
                    self._spawn_npc(e, ni)

        # ---- reward = delta(snake_length) ----
        reward = <double>(self.plength[e] - length_before)
        self.rew_buf[e] = <float>reward
        self.ep_return[e] += reward
        self.ep_len[e] = self.pstep[e]

        # ---- check termination ----
        cdef int truncated = 1 if self.pstep[e] >= self.max_steps else 0
        if died or truncated:
            self.done_buf[e] = 1
            self.ep_ret_buf[e]  = <float>self.ep_return[e]
            self.ep_len_buf[e]  = self.pstep[e]
            self.ep_slen_buf[e] = self.plength[e]
            self._reset_env(e)

    # -------------------------------------------------------- NPC AI
    cdef void _step_npcs(self, int e) noexcept nogil:
        cdef int npc_base = e * NUM_NPCS
        cdef int ni, nidx, fi, nearest_fi, new_head
        cdef double dx, dy, dist_sq, min_dist, desired, diff
        cdef unsigned long long* rng = &self.rng_state[e]

        for ni in range(NUM_NPCS):
            nidx = npc_base + ni
            if not self.nalive[nidx]:
                continue

            # Find nearest food
            min_dist = 1e30
            nearest_fi = -1
            for fi in range(MAX_FOOD):
                if not self.factive[e, fi]:
                    continue
                dx = self.fx[e, fi] - self.nx[nidx]
                dy = self.fy[e, fi] - self.ny[nidx]
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    nearest_fi = fi

            if nearest_fi >= 0:
                dx = self.fx[e, nearest_fi] - self.nx[nidx]
                dy = self.fy[e, nearest_fi] - self.ny[nidx]
                desired = atan2(dy, dx)
                diff = desired - self.ndir[nidx]
                # Normalise to [-pi, pi]
                while diff > M_PI:
                    diff -= 2.0 * M_PI
                while diff < -M_PI:
                    diff += 2.0 * M_PI
                if diff > self.turn_rate:
                    self.ndir[nidx] += self.turn_rate
                elif diff < -self.turn_rate:
                    self.ndir[nidx] -= self.turn_rate
                else:
                    self.ndir[nidx] = desired

            # Random noise
            if _rand_double(rng) < 0.15:
                self.ndir[nidx] += (_rand_double(rng) - 0.5) * 1.0

            # Normalise direction
            if self.ndir[nidx] > M_PI:
                self.ndir[nidx] -= 2.0 * M_PI
            if self.ndir[nidx] < -M_PI:
                self.ndir[nidx] += 2.0 * M_PI

            # Move
            self.nx[nidx] += cos(self.ndir[nidx]) * self.base_speed
            self.ny[nidx] += sin(self.ndir[nidx]) * self.base_speed
            new_head = (self.nseg_head[nidx] + 1) % MAX_SEG
            self.nseg_head[nidx] = new_head
            self.nseg_x[nidx, new_head] = self.nx[nidx]
            self.nseg_y[nidx, new_head] = self.ny[nidx]

    # -------------------------------------------------------- NPC eats food
    cdef void _npc_eat_food(self, int e) noexcept nogil:
        cdef int npc_base = e * NUM_NPCS
        cdef int ni, nidx, fi
        cdef double dx, dy, dist_sq
        cdef double eat_r_sq = (self.head_radius + self.food_radius) * (self.head_radius + self.food_radius)

        for ni in range(NUM_NPCS):
            nidx = npc_base + ni
            if not self.nalive[nidx]:
                continue
            for fi in range(MAX_FOOD):
                if not self.factive[e, fi]:
                    continue
                dx = self.fx[e, fi] - self.nx[nidx]
                dy = self.fy[e, fi] - self.ny[nidx]
                dist_sq = dx * dx + dy * dy
                if dist_sq < eat_r_sq:
                    self.factive[e, fi] = 0
                    if self.nlength[nidx] < MAX_SEG:
                        self.nlength[nidx] += 1

    # ------------------------------------------------ collision helpers
    cdef int _player_hits_npc_body(self, int e) noexcept nogil:
        """Returns 1 if player head hit any NPC body segment."""
        cdef int npc_base = e * NUM_NPCS
        cdef int ni, nidx, k, seg_idx
        cdef double dx, dy, dist_sq
        cdef double thr_sq = (self.head_radius + self.body_radius) * (self.head_radius + self.body_radius)

        for ni in range(NUM_NPCS):
            nidx = npc_base + ni
            if not self.nalive[nidx]:
                continue
            if self.nlength[nidx] <= 3:
                continue
            for k in range(3, self.nlength[nidx]):
                seg_idx = (self.nseg_head[nidx] - k + MAX_SEG) % MAX_SEG
                dx = self.nseg_x[nidx, seg_idx] - self.px[e]
                dy = self.nseg_y[nidx, seg_idx] - self.py[e]
                dist_sq = dx * dx + dy * dy
                if dist_sq < thr_sq:
                    return 1
        return 0

    cdef int _npc_hits_player_body(self, int e) noexcept nogil:
        """Returns number of NPCs killed by running into the player's body."""
        cdef int npc_base = e * NUM_NPCS
        cdef int ni, nidx, k, seg_idx, kills = 0
        cdef double dx, dy, dist_sq
        cdef double thr_sq = (self.head_radius + self.body_radius) * (self.head_radius + self.body_radius)

        if self.plength[e] <= 3:
            return 0
        for ni in range(NUM_NPCS):
            nidx = npc_base + ni
            if not self.nalive[nidx]:
                continue
            for k in range(3, self.plength[e]):
                seg_idx = (self.pseg_head[e] - k + MAX_SEG) % MAX_SEG
                dx = self.pseg_x[e, seg_idx] - self.nx[nidx]
                dy = self.pseg_y[e, seg_idx] - self.ny[nidx]
                dist_sq = dx * dx + dy * dy
                if dist_sq < thr_sq:
                    self.nalive[nidx] = 0
                    self.nrespawn[nidx] = self.npc_respawn_delay
                    kills += 1
                    break
        return kills

    cdef void _npc_boundary_check(self, int e) noexcept nogil:
        cdef int npc_base = e * NUM_NPCS
        cdef int ni, nidx
        cdef double dist_sq
        cdef double ar_sq = self.arena_radius * self.arena_radius

        for ni in range(NUM_NPCS):
            nidx = npc_base + ni
            if not self.nalive[nidx]:
                continue
            dist_sq = self.nx[nidx] * self.nx[nidx] + self.ny[nidx] * self.ny[nidx]
            if dist_sq > ar_sq:
                self.nalive[nidx] = 0
                self.nrespawn[nidx] = self.npc_respawn_delay

    # ------------------------------------------------- observation
    cdef void _compute_obs_env(self, int e) noexcept nogil:
        cdef double cd = cos(self.pdir[e])
        cdef double sd = sin(self.pdir[e])
        cdef double inv_vp = 1.0 / self.viewport
        cdef double inv_ar = 1.0 / self.arena_radius
        cdef double inv_ml = 1.0 / <double>MAX_SEG
        cdef int i, j, k, seg_idx, nidx
        cdef double dx, dy, ego_x, ego_y, dist_sq

        # --- player state (6) ---
        self.obs_buf[e, 0] = <float>(<double>self.plength[e] * inv_ml)
        self.obs_buf[e, 1] = <float>(1.0 if self.pboosting[e] else 0.5)
        self.obs_buf[e, 2] = <float>sd
        self.obs_buf[e, 3] = <float>cd
        self.obs_buf[e, 4] = <float>(sqrt(self.px[e]*self.px[e] + self.py[e]*self.py[e]) * inv_ar)
        self.obs_buf[e, 5] = <float>self.pboosting[e]

        # --- K nearest foods (K_FOOD * 2) ---
        cdef double fd[K_FOOD]
        cdef double fex[K_FOOD]
        cdef double fey[K_FOOD]
        cdef double max_fd
        cdef int max_fi

        for i in range(K_FOOD):
            fd[i]  = 1e30
            fex[i] = 0.0
            fey[i] = 0.0
        max_fd = 1e30
        max_fi = 0

        for i in range(MAX_FOOD):
            if not self.factive[e, i]:
                continue
            dx = self.fx[e, i] - self.px[e]
            dy = self.fy[e, i] - self.py[e]
            dist_sq = dx * dx + dy * dy
            if dist_sq < max_fd:
                ego_x =  cd * dx + sd * dy
                ego_y = -sd * dx + cd * dy
                fd[max_fi]  = dist_sq
                fex[max_fi] = ego_x * inv_vp
                fey[max_fi] = ego_y * inv_vp
                # find new worst slot
                max_fd = fd[0]
                max_fi = 0
                for j in range(1, K_FOOD):
                    if fd[j] > max_fd:
                        max_fd = fd[j]
                        max_fi = j

        for i in range(K_FOOD):
            if fd[i] < 1e29:
                self.obs_buf[e, 6 + i * 2]     = <float>fex[i]
                self.obs_buf[e, 6 + i * 2 + 1] = <float>fey[i]
            else:
                self.obs_buf[e, 6 + i * 2]     = 0.0
                self.obs_buf[e, 6 + i * 2 + 1] = 0.0

        # --- K nearest NPC segments (K_NPC * 2) ---
        cdef double nd[K_NPC]
        cdef double nex[K_NPC]
        cdef double ney[K_NPC]
        cdef double max_nd
        cdef int max_ni

        for i in range(K_NPC):
            nd[i]  = 1e30
            nex[i] = 0.0
            ney[i] = 0.0
        max_nd = 1e30
        max_ni = 0

        cdef int npc_base = e * NUM_NPCS
        for i in range(NUM_NPCS):
            nidx = npc_base + i
            if not self.nalive[nidx]:
                continue
            for k in range(self.nlength[nidx]):
                seg_idx = (self.nseg_head[nidx] - k + MAX_SEG) % MAX_SEG
                dx = self.nseg_x[nidx, seg_idx] - self.px[e]
                dy = self.nseg_y[nidx, seg_idx] - self.py[e]
                dist_sq = dx * dx + dy * dy
                if dist_sq < max_nd:
                    ego_x =  cd * dx + sd * dy
                    ego_y = -sd * dx + cd * dy
                    nd[max_ni]  = dist_sq
                    nex[max_ni] = ego_x * inv_vp
                    ney[max_ni] = ego_y * inv_vp
                    max_nd = nd[0]
                    max_ni = 0
                    for j in range(1, K_NPC):
                        if nd[j] > max_nd:
                            max_nd = nd[j]
                            max_ni = j

        cdef int off = 6 + K_FOOD * 2  # = 38
        for i in range(K_NPC):
            if nd[i] < 1e29:
                self.obs_buf[e, off + i * 2]     = <float>nex[i]
                self.obs_buf[e, off + i * 2 + 1] = <float>ney[i]
            else:
                self.obs_buf[e, off + i * 2]     = 0.0
                self.obs_buf[e, off + i * 2 + 1] = 0.0
