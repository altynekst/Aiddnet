#include "ai_controller.h"
#include "ai_math.h"
#include <cmath>
#include <algorithm>

namespace AI
{

AIController::AIController(AINeuralNetwork* pNeuralNet, AIKDTreeMemory* pKDTree) :
    m_pNeuralNet(pNeuralNet),
    m_pKDTree(pKDTree)
{
    Reset();
}

void AIController::Reset()
{
    m_LastAction = AIAction();
    m_CurrentAimX = 100.0f;
    m_CurrentAimY = 0.0f;
    m_JumpCooldown = 0;
    m_HookHoldTicks = 0;
    m_HookCycleTimer = 0;
    m_HammerCooldown = 0;
    m_StrafeTimer = 0;
    m_StrafeDir = 0;
}

AIAction AIController::Step(const AIFeatures& current_state, float delta_time)
{
    if(m_Mode == EAIMode::OFF || !m_pNeuralNet)
    {
        return AIAction();
    }

    // Decrement internal weapon reload timer
    if(m_HammerCooldown > 0)
    {
        m_HammerCooldown--;
    }

    // 1. Neural Network Prediction (Baseline Imitation)
    AIAction nn_action = m_pNeuralNet->Predict(current_state);

    // 2. KD-Tree Memory Prediction (K-Nearest Neighbors)
    AIAction knn_action;
    bool has_knn = false;
    if(m_pKDTree && m_KNNBlendWeight > 0.01f)
    {
        has_knn = m_pKDTree->PredictAction(current_state, knn_action, 5);
    }

    AIAction final_action = nn_action;

    if(has_knn)
    {
        float w_knn = m_KNNBlendWeight;
        float w_nn = 1.0f - w_knn;

        final_action.m_TargetX = w_nn * nn_action.m_TargetX + w_knn * knn_action.m_TargetX;
        final_action.m_TargetY = w_nn * nn_action.m_TargetY + w_knn * knn_action.m_TargetY;

        if(knn_action.m_Confidence > 0.7f && knn_action.m_Direction != 0)
        {
            final_action.m_Direction = knn_action.m_Direction;
        }

        if(knn_action.m_Hook) final_action.m_Hook = true;
        if(knn_action.m_Jump && m_JumpCooldown <= 0) final_action.m_Jump = true;
    }

    // Default weapon fire and tactics
    final_action.m_Fire = false;
    final_action.m_Tactics = EBlockTactics::NEUTRAL_SPACING;

    float desired_aim_x = final_action.m_TargetX;
    float desired_aim_y = final_action.m_TargetY;
    bool fast_flick_aim = false;

    // 3. COMBAT GEOMETRY & TARGETING
    const bool InDanger = (current_state.m_InDangerZone > 0.5f || current_state.m_HazardDistSelf < 0.15f);
    const bool EnemyNear = (current_state.m_EnemyDist < 0.95f);
    const bool EnemyHooked = (current_state.m_HookedPlayer > 0.5f);

    float enemy_dx = current_state.m_EnemyRelX * MAX_DIST_SCALE;
    float enemy_dy = current_state.m_EnemyRelY * MAX_DIST_SCALE;
    float enemy_vx = current_state.m_EnemyRelVelX * MAX_VEL_SCALE;
    float enemy_vy = current_state.m_EnemyRelVelY * MAX_VEL_SCALE;
    float enemy_dist = current_state.m_EnemyDist * MAX_DIST_SCALE;

    float self_vx = current_state.m_VelX * MAX_VEL_SCALE;
    float self_vy = current_state.m_VelY * MAX_VEL_SCALE;

    // Freeze state discrimination:
    // 0.0f = Unfrozen (active threat)
    // > 0.05f = Frozen (shallow or deep freeze -> NEVER unfreeze with hammer!)
    const bool EnemyIsFrozen = (EnemyNear && current_state.m_EnemyFrozen > 0.05f);
    const bool EnemyActive = (EnemyNear && !EnemyIsFrozen);
    const bool EnemyDeepFrozen = (EnemyNear && current_state.m_EnemyFrozen >= 0.8f);
    // True bottom pit / permanent incapacitated freeze check (hands-off zone):
    // Enemy is in the bottom hazard pit, at or below bot level, and cannot recover
    const bool EnemyInBottomFreezePit = (EnemyDeepFrozen || 
        (current_state.m_EnemyHazardDist < 0.18f && enemy_dy >= -10.0f && current_state.m_HazardDistSelf < 0.45f));

    // Relative vertical stacking checks (Teeworlds coordinates: +Y is down, -Y is up)
    // If enemy_dy > 6: Enemy is below self -> Self is ON ENEMY'S HEAD
    // If enemy_dy < -6: Enemy is above self -> Self is UNDER ENEMY
    const bool OnEnemyHead = (EnemyNear && std::abs(enemy_dx) < 36.0f && enemy_dy > 6.0f && enemy_dy < 64.0f);
    const bool DirectlyUnderEnemy = (EnemyNear && std::abs(enemy_dx) < 36.0f && enemy_dy < -6.0f && enemy_dy > -64.0f);

    // Aim Alignment & Relative Velocity calculations
    float enemy_dir_x = (enemy_dist > 0.001f) ? (enemy_dx / enemy_dist) : 1.0f;
    float enemy_dir_y = (enemy_dist > 0.001f) ? (enemy_dy / enemy_dist) : 0.0f;

    // Relative velocity: negative indicates closing distance rapidly towards collision
    float rel_approach_vel = (enemy_vx - self_vx) * enemy_dir_x + (enemy_vy - self_vy) * enemy_dir_y;
    bool is_closing_in = (rel_approach_vel < -25.0f || (EnemyHooked && enemy_dist < 180.0f));

    // True hammer melee reach threshold (Teeworlds hammer radius is ~70-84 px)
    const bool InHammerRange = (EnemyNear && enemy_dist <= 84.0f);

    // Stationary opponent detection (AFK / non-moving player)
    const bool EnemyIsStationary = (EnemyActive && std::abs(enemy_vx) < 30.0f && std::abs(enemy_vy) < 45.0f);

    // Knockback alignment with freeze hazard:
    // Knock vector points from self to enemy (enemy_dx, enemy_dy).
    // Does this vector send the enemy towards the hazard or off a platform?
    float knock_dir_x = (enemy_dx >= 0.0f) ? 1.0f : -1.0f;
    bool knock_towards_hazard = false;
    if(current_state.m_HazardDirSelfX != 0.0f)
    {
        knock_towards_hazard = (knock_dir_x * current_state.m_HazardDirSelfX > 0.0f);
    }
    else if(current_state.m_EnemyHazardDist < 0.35f)
    {
        knock_towards_hazard = true;
    }

    // High Ground / Y-Block detection:
    const bool EnemyOnHighGround = (EnemyNear && enemy_dy < -25.0f && enemy_dist < 450.0f);
    const bool EnemyBelowLedge = (EnemyNear && enemy_dy > 25.0f && enemy_dist < 450.0f);

    // Airborne / High Launch condition for opponent:
    bool enemy_airborne = (enemy_dy < -12.0f || enemy_vy < -100.0f || current_state.m_IsGrounded < 0.5f);

    // Helper: is moving in direction 'dir' heading directly into a freeze hazard?
    auto IsDirectionTowardsHazard = [&](int dir) -> bool {
        if(dir == 0) return false;
        if(current_state.m_HazardDistSelf > 0.40f) return false; // Safe distance
        if(dir > 0 && current_state.m_HazardDirSelfX > 0.15f) return true; // Freeze is to the right
        if(dir < 0 && current_state.m_HazardDirSelfX < -0.15f) return true; // Freeze is to the left
        return false;
    };

    // =========================================================================
    // PRIORITY 1: Emergency Hazard Recovery (imminent freeze danger for self)
    // =========================================================================
    if(m_FreezeAvoidanceEnabled && InDanger)
    {
        final_action.m_Tactics = EBlockTactics::DEFENSIVE_RECOVERY;

        // Counter Steer strictly away from hazard
        if(current_state.m_HazardDirSelfX > 0.05f)
            final_action.m_Direction = -1;
        else if(current_state.m_HazardDirSelfX < -0.05f)
            final_action.m_Direction = 1;

        // Emergency Ceiling / Wall Anchor Hook
        float best_anchor_score = -1e9f;
        int best_ray_idx = -1;
        for(int r = 0; r < NUM_RAYCASTS; ++r)
        {
            float wall_d = current_state.m_aWallDist[r];
            if(wall_d < 0.95f)
            {
                float ray_angle = (float)r * (2.0f * 3.14159265f / (float)NUM_RAYCASTS);
                float rx = std::cos(ray_angle);
                float ry = std::sin(ray_angle);

                float upward_bonus = (ry < 0.0f) ? (-ry * 3.0f) : -1.0f;
                float score = upward_bonus + (1.0f - wall_d);

                if(score > best_anchor_score)
                {
                    best_anchor_score = score;
                    best_ray_idx = r;
                }
            }
        }

        if(best_ray_idx >= 0 && m_SmartHookEnabled)
        {
            float ray_angle = (float)best_ray_idx * (2.0f * 3.14159265f / (float)NUM_RAYCASTS);
            desired_aim_x = std::cos(ray_angle) * 350.0f;
            desired_aim_y = std::sin(ray_angle) * 350.0f;
            final_action.m_Hook = true;
            fast_flick_aim = true;
        }

        if(current_state.m_IsGrounded > 0.5f || current_state.m_CanDoubleJump > 0.5f)
        {
            final_action.m_Jump = true;
        }

        // Defensive Self-Rescue Hammer Blast (ONLY on active enemy to clear escape space)
        if(EnemyActive && InHammerRange && m_HammerCooldown <= 0)
        {
            desired_aim_x = enemy_dx;
            desired_aim_y = enemy_dy;
            fast_flick_aim = true;
            final_action.m_WantedWeapon = 0; // Hammer
            final_action.m_Fire = true;
            m_HammerCooldown = 24;
            final_action.m_Hook = false;
        }
    }
    // =========================================================================
    // PRIORITY 2: Anti-Head-Stacking (Self is on top of enemy head)
    // =========================================================================
    else if(OnEnemyHead)
    {
        final_action.m_Tactics = EBlockTactics::HAMMER_EXECUTE;

        // Slide sideways off enemy head away from hazard
        int slide_dir = (enemy_dx >= 0.0f) ? 1 : -1;
        if(IsDirectionTowardsHazard(slide_dir))
            slide_dir = -slide_dir;

        final_action.m_Direction = slide_dir;
        final_action.m_Jump = true; // Hop off

        // Hammer downwards to slam enemy down and propel self up/sideways (only if active)
        desired_aim_x = (enemy_dx >= 0.0f) ? 20.0f : -20.0f;
        desired_aim_y = 220.0f;
        fast_flick_aim = true;
        final_action.m_WantedWeapon = 0; // Hammer

        if(EnemyActive && m_HammerCooldown <= 0)
        {
            final_action.m_Fire = true;
            m_HammerCooldown = 24;
            final_action.m_Hook = false;
        }
    }
    // =========================================================================
    // PRIORITY 3: Anti-Under-Enemy (Enemy is on top of self)
    // =========================================================================
    else if(DirectlyUnderEnemy)
    {
        final_action.m_Tactics = EBlockTactics::HAMMER_EXECUTE;

        // Step OUT from underneath the enemy (opposite to stack center, away from hazard)
        int step_dir = (enemy_dx >= 0.0f) ? -1 : 1;
        if(IsDirectionTowardsHazard(step_dir))
            step_dir = -step_dir;

        final_action.m_Direction = step_dir;

        // Aim up at the enemy
        desired_aim_x = enemy_dx;
        desired_aim_y = -220.0f;
        fast_flick_aim = true;
        final_action.m_WantedWeapon = 0; // Hammer

        if(current_state.m_IsGrounded > 0.5f || current_state.m_CanDoubleJump > 0.5f)
        {
            final_action.m_Jump = true; // Jump out to equal level
        }

        if(EnemyActive && m_HammerCooldown <= 0)
        {
            final_action.m_Fire = true;
            m_HammerCooldown = 24;
            final_action.m_Hook = false;
        }
    }
    // =========================================================================
    // PRIORITY 4: FROZEN ENEMY HANDLING & Y-BLOCK EJECTION SCENARIOS
    // =========================================================================
    else if(m_BlockMode && EnemyIsFrozen)
    {
        final_action.m_Tactics = EBlockTactics::FREEZE_SLAM;
        final_action.m_Fire = false;

        // ---------------------------------------------------------------------
        // SCENARIO 4A: ENEMY IN BOTTOM PIT / PERMANENT DEEP FREEZE
        // (Hands-off! Do NOT hook out of freeze! Do NOT hit with hammer!)
        // ---------------------------------------------------------------------
        if(EnemyInBottomFreezePit)
        {
            final_action.m_Hook = false; // NEVER hook a player in the pit!
            final_action.m_Fire = false; // NEVER strike a player in the pit!

            // Stand guard outside the pit on safe ground
            if(current_state.m_HazardDistSelf < 0.35f && current_state.m_HazardDirSelfX != 0.0f)
            {
                // Step away from freeze edge back onto dry land
                final_action.m_Direction = (current_state.m_HazardDirSelfX > 0.0f) ? -1 : 1;
            }
            else
            {
                final_action.m_Direction = 0; // Hold safe guard position
            }

            // Aim towards enemy or upward to watch the pit
            desired_aim_x = enemy_dx;
            desired_aim_y = enemy_dy;
        }
        // ---------------------------------------------------------------------
        // SCENARIO 4B: ENEMY FROZEN ON Y-BLOCK / HIGH GROUND ABOVE BOT
        // (Enemy is frozen on upper platform -> HOOK THEM DOWN OFF THE Y-BLOCK!)
        // ---------------------------------------------------------------------
        else if(EnemyOnHighGround || enemy_dy < -15.0f)
        {
            // Aim straight at the frozen player on the Y-block
            desired_aim_x = enemy_dx;
            desired_aim_y = enemy_dy;
            fast_flick_aim = true;

            // Throw hook to yank them off the platform ledge!
            final_action.m_Hook = true;

            // Step slightly sideways away from stack so the hook angle pulls them over the platform lip
            if(std::abs(enemy_dx) < 40.0f)
            {
                // Move towards nearest open side
                int open_dir = (current_state.m_aWallDist[0] > current_state.m_aWallDist[8]) ? 1 : -1;
                if(IsDirectionTowardsHazard(open_dir)) open_dir = -open_dir;
                final_action.m_Direction = open_dir;
            }
            else
            {
                final_action.m_Direction = (enemy_dx >= 0.0f) ? 1 : -1;
                if(IsDirectionTowardsHazard(final_action.m_Direction))
                    final_action.m_Direction = 0;
            }

            // If enemy is within hammer reach on their way down, strike them into the pit!
            if(InHammerRange && m_FreezeFinisherEnabled)
            {
                desired_aim_x = enemy_dx;
                desired_aim_y = enemy_dy + 15.0f;
                final_action.m_WantedWeapon = 0;
                if(m_HammerCooldown <= 0)
                {
                    final_action.m_Fire = true;
                    m_HammerCooldown = 22;
                    final_action.m_Hook = false; // Release hook for maximum downward fling
                }
            }
        }
        // ---------------------------------------------------------------------
        // SCENARIO 4C: BOT AND FROZEN ENEMY ARE ON THE SAME Y-BLOCK / PLATFORM
        // (Slam frozen enemy OFF the platform edge into the drop/hazard below!)
        // ---------------------------------------------------------------------
        else if(std::abs(enemy_dy) <= 25.0f && (EnemyBelowLedge || current_state.m_aWallDist[4] > 0.15f || current_state.m_EnemyHazardDist < 0.65f))
        {
            if(InHammerRange && m_FreezeFinisherEnabled)
            {
                // Hammer strike angled downwards towards the drop
                desired_aim_x = enemy_dx;
                desired_aim_y = enemy_dy + 15.0f;
                fast_flick_aim = true;
                final_action.m_WantedWeapon = 0; // Hammer

                if(m_HammerCooldown <= 0)
                {
                    final_action.m_Fire = true; // Slam off the Y-block platform!
                    m_HammerCooldown = 22;
                    final_action.m_Hook = false; // Release hook so knockback sends them flying off!
                    if(current_state.m_IsGrounded > 0.5f && current_state.m_HazardDistSelf > 0.35f)
                    {
                        final_action.m_Jump = true; // Hop into strike for extra momentum
                    }
                }
            }
            else
            {
                // Hook the frozen enemy to drag them towards the edge and close distance
                desired_aim_x = enemy_dx;
                desired_aim_y = enemy_dy;
                final_action.m_Hook = true;

                // Move towards the enemy to get in hammer range (45-70px)
                int move_dir = (enemy_dx >= 0.0f) ? 1 : -1;
                if(!IsDirectionTowardsHazard(move_dir))
                    final_action.m_Direction = move_dir;
                else
                    final_action.m_Direction = 0;
            }
        }
        // ---------------------------------------------------------------------
        // SCENARIO 4D: ENEMY FROZEN ON SAFE FLAT GROUND
        // (Drag or hammer towards the nearest freeze hazard)
        // ---------------------------------------------------------------------
        else
        {
            int hazard_dir = (current_state.m_HazardDirSelfX > 0.0f) ? 1 : (current_state.m_HazardDirSelfX < 0.0f ? -1 : 0);

            // If hazard exists:
            if(hazard_dir != 0)
            {
                // Bot is between enemy and hazard (hazard is behind bot):
                // Hook the frozen enemy to DRAG them into the hazard!
                if((enemy_dx > 0 && hazard_dir < 0) || (enemy_dx < 0 && hazard_dir > 0))
                {
                    desired_aim_x = enemy_dx;
                    desired_aim_y = enemy_dy;
                    final_action.m_Hook = true;

                    // Stand safely at the edge, don't fall in
                    if(current_state.m_HazardDistSelf < 0.35f)
                        final_action.m_Direction = -hazard_dir;
                    else
                        final_action.m_Direction = 0;
                }
                // Bot is BEHIND the enemy relative to hazard (Bot -> Enemy -> Hazard):
                // Push the enemy with hammer into the hazard!
                else
                {
                    if(InHammerRange && m_FreezeFinisherEnabled)
                    {
                        desired_aim_x = enemy_dx;
                        desired_aim_y = enemy_dy + 10.0f;
                        fast_flick_aim = true;
                        final_action.m_WantedWeapon = 0;

                        if(m_HammerCooldown <= 0)
                        {
                            final_action.m_Fire = true;
                            m_HammerCooldown = 22;
                            final_action.m_Hook = false;
                        }
                    }
                    else
                    {
                        final_action.m_Direction = (enemy_dx > 0.0f) ? 1 : -1;
                        if(IsDirectionTowardsHazard(final_action.m_Direction))
                            final_action.m_Direction = 0;
                        desired_aim_x = enemy_dx;
                        desired_aim_y = enemy_dy;
                        final_action.m_Hook = (enemy_dist > 80.0f);
                    }
                }
            }
            else
            {
                // No clear hazard direction: hook and drag towards open space
                if(enemy_dist < 400.0f && m_SmartHookEnabled)
                {
                    desired_aim_x = enemy_dx;
                    desired_aim_y = enemy_dy;
                    final_action.m_Hook = true;
                    final_action.m_Direction = (enemy_dx > 0.0f) ? -1 : 1;
                }
            }
        }
    }
    // =========================================================================
    // PRIORITY 5: HIGH GROUND ENGAGEMENT (Enemy is above on Y-Block / High Ledge)
    // =========================================================================
    else if(m_BlockMode && EnemyActive && EnemyOnHighGround)
    {
        final_action.m_Tactics = EBlockTactics::OFFENSIVE_DRAG;

        // Move laterally towards the climbable corner/edge of the high block
        float climb_dir_x = (enemy_dx >= 0.0f) ? 1.0f : -1.0f;
        if(IsDirectionTowardsHazard((int)climb_dir_x))
        {
            climb_dir_x = -climb_dir_x;
        }
        final_action.m_Direction = (climb_dir_x > 0.0f) ? 1 : -1;

        // Jump mechanics for climbing:
        if(current_state.m_IsGrounded > 0.5f)
        {
            final_action.m_Jump = true; // First jump from ground
        }
        else if(current_state.m_CanDoubleJump > 0.5f && self_vy >= -120.0f)
        {
            final_action.m_Jump = true; // Double Jump at apex to reach platform height!
        }

        // Hook Navigation:
        // DO NOT hook the enemy if directly underneath! Hook the wall/corner above to hoist onto platform!
        int best_hoist_ray = -1;
        float best_hoist_dist = 1.0f;
        int hoist_rays[] = { (climb_dir_x > 0 ? 14 : 10), 12, (climb_dir_x > 0 ? 15 : 9), (climb_dir_x > 0 ? 13 : 11) };
        for(int r : hoist_rays)
        {
            if(current_state.m_aWallDist[r] < 0.85f && current_state.m_aWallDist[r] < best_hoist_dist)
            {
                best_hoist_dist = current_state.m_aWallDist[r];
                best_hoist_ray = r;
            }
        }

        if(best_hoist_ray >= 0 && m_SmartHookEnabled && current_state.m_IsGrounded < 0.5f)
        {
            float ray_angle = (float)best_hoist_ray * (2.0f * 3.14159265f / (float)NUM_RAYCASTS);
            desired_aim_x = std::cos(ray_angle) * 350.0f;
            desired_aim_y = std::sin(ray_angle) * 350.0f;
            final_action.m_Hook = true;
            fast_flick_aim = true;
        }
        else if(std::abs(enemy_dx) > 60.0f && m_SmartHookEnabled && enemy_dist < 320.0f)
        {
            // If at a clear angle from the side (not straight under), hook enemy to sling up
            desired_aim_x = enemy_dx;
            desired_aim_y = enemy_dy;
            final_action.m_Hook = true;
        }
        else
        {
            // Do not hook straight up into ceiling! Keep aiming up-forward to navigate
            desired_aim_x = climb_dir_x * 120.0f;
            desired_aim_y = -200.0f;
            final_action.m_Hook = false;
        }
    }
    // =========================================================================
    // PRIORITY 6: TACTICAL STRIKE SCENARIOS FOR ACTIVE OPPONENTS
    // =========================================================================
    else if(m_BlockMode && EnemyActive)
    {
        // ---------------------------------------------------------------------
        // SCENARIO 6-STATIONARY: STATIONARY / AFK OPPONENT BLOCK & HAZARD DRAG
        // (Active enemy not moving -> Hook tether, reposition behind, sweetspot strike)
        // ---------------------------------------------------------------------
        if(EnemyIsStationary && !EnemyOnHighGround)
        {
            final_action.m_Tactics = EBlockTactics::OFFENSIVE_DRAG;
            desired_aim_x = enemy_dx;
            desired_aim_y = enemy_dy;
            fast_flick_aim = true;
            final_action.m_WantedWeapon = 0; // Hammer

            int hazard_dir = (current_state.m_HazardDirSelfX > 0.0f) ? 1 : (current_state.m_HazardDirSelfX < 0.0f ? -1 : 0);

            // If freeze hazard is known:
            if(hazard_dir != 0)
            {
                // Case 1: Bot is on safe side, enemy is between bot and hazard (Bot -> Enemy -> Hazard)
                // Position is PERFECT for hammer strike into hazard!
                if((enemy_dx > 0 && hazard_dir > 0) || (enemy_dx < 0 && hazard_dir < 0))
                {
                    // Move into close sweetspot (45-72px)
                    if(enemy_dist > 65.0f)
                    {
                        final_action.m_Direction = (enemy_dx > 0.0f) ? 1 : -1;
                    }
                    else if(enemy_dist < 40.0f)
                    {
                        final_action.m_Direction = (enemy_dx > 0.0f) ? -1 : 1;
                    }
                    else
                    {
                        final_action.m_Direction = 0;
                    }

                    // Tether with hook
                    final_action.m_Hook = true;

                    // Sweetspot strike range: 40-75px
                    if(enemy_dist <= 75.0f && m_HammerCooldown <= 0)
                    {
                        final_action.m_Fire = true;
                        m_HammerCooldown = 22;
                        final_action.m_Hook = false; // Release hook so knockback launches full velocity!
                        if(current_state.m_IsGrounded > 0.5f && current_state.m_HazardDistSelf > 0.35f)
                        {
                            final_action.m_Jump = true; // Forward hop for extra kinetic push
                        }
                    }
                }
                // Case 2: Bot is between enemy and hazard (Hazard <- Bot <- Enemy)
                // Bot is blocking the hazard entrance! Hook enemy and walk/jump past to get behind them!
                else
                {
                    final_action.m_Hook = true; // Hook the stationary enemy to pull/drag
                    // Move towards the enemy to pass them or drag them towards hazard
                    final_action.m_Direction = (enemy_dx > 0.0f) ? 1 : -1;
                    if(enemy_dist < 75.0f && current_state.m_IsGrounded > 0.5f)
                    {
                        final_action.m_Jump = true; // Hop over them to safe side
                    }
                }
            }
            else
            {
                // No hazard direction: hook tether and strike when in sweetspot
                final_action.m_Hook = true;
                if(enemy_dist > 65.0f)
                    final_action.m_Direction = (enemy_dx > 0.0f) ? 1 : -1;
                else if(enemy_dist < 40.0f)
                    final_action.m_Direction = (enemy_dx > 0.0f) ? -1 : 1;
                else
                    final_action.m_Direction = 0;

                if(enemy_dist <= 75.0f && m_HammerCooldown <= 0)
                {
                    final_action.m_Fire = true;
                    m_HammerCooldown = 22;
                    final_action.m_Hook = false;
                }
            }
        }
        // ---------------------------------------------------------------------
        // SCENARIO 6A: FREEZE EJECTION STRIKE (Knocks enemy into Freeze Hazard)
        // ---------------------------------------------------------------------
        else if(InHammerRange && knock_towards_hazard && m_FreezeFinisherEnabled)
        {
            final_action.m_Tactics = EBlockTactics::FREEZE_SLAM;
            desired_aim_x = enemy_dx;
            desired_aim_y = enemy_dy + 15.0f;
            fast_flick_aim = true;
            final_action.m_WantedWeapon = 0; // Hammer

            // Hold ground / counter-steer away from freeze so bot does NOT fall in!
            if(current_state.m_HazardDistSelf < 0.35f && current_state.m_HazardDirSelfX != 0.0f)
            {
                final_action.m_Direction = (current_state.m_HazardDirSelfX > 0.0f) ? -1 : 1;
            }

            if(m_HammerCooldown <= 0)
            {
                final_action.m_Fire = true;
                m_HammerCooldown = 24;
                final_action.m_Hook = false;
                // NEVER jump forward into freeze! Only jump if safely on dry land away from edge
                if(current_state.m_IsGrounded > 0.5f && current_state.m_HazardDistSelf > 0.40f)
                {
                    final_action.m_Jump = true;
                }
            }
        }
        // ---------------------------------------------------------------------
        // SCENARIO 6B: FLY-IN MOMENTUM STRIKE ("Залётный удар")
        // ---------------------------------------------------------------------
        else if(InHammerRange && m_FlyStrikeEnabled && (is_closing_in || EnemyHooked || m_HookHoldTicks > 0))
        {
            final_action.m_Tactics = EBlockTactics::FLY_STRIKE;

            desired_aim_x = enemy_dx * 2.0f;
            desired_aim_y = enemy_dy * 2.0f - 6.0f;
            fast_flick_aim = true;
            final_action.m_WantedWeapon = 0; // Hammer

            float strike_dir_x = (enemy_dx >= 0.0f) ? 1.0f : -1.0f;
            if(current_state.m_HazardDirSelfX != 0.0f)
            {
                strike_dir_x = (current_state.m_HazardDirSelfX > 0.0f) ? 1.0f : -1.0f;
            }
            if(IsDirectionTowardsHazard((int)strike_dir_x))
            {
                strike_dir_x = -strike_dir_x; // Never charge towards freeze!
            }
            final_action.m_Direction = (strike_dir_x > 0.0f) ? 1 : -1;

            if(m_HammerCooldown <= 0)
            {
                final_action.m_Fire = true;
                m_HammerCooldown = 24;
                final_action.m_Hook = false; // RELEASE HOOK SO ENEMY FLIES FREELY!
                if((current_state.m_IsGrounded > 0.5f || enemy_dy < 0.0f) && current_state.m_HazardDistSelf > 0.35f)
                {
                    final_action.m_Jump = true;
                }
            }
        }
        // ---------------------------------------------------------------------
        // SCENARIO 6C: SKY LAUNCH / KINETIC AIR INTERCEPT (Enemy airborne or rushing)
        // ---------------------------------------------------------------------
        else if(InHammerRange && (enemy_airborne || rel_approach_vel < -30.0f))
        {
            final_action.m_Tactics = EBlockTactics::HAMMER_EXECUTE;

            // Aim upwards/towards enemy to launch them sky-high
            desired_aim_x = enemy_dx;
            desired_aim_y = enemy_dy - 12.0f;
            fast_flick_aim = true;
            final_action.m_WantedWeapon = 0; // Hammer

            if(m_HammerCooldown <= 0)
            {
                final_action.m_Fire = true;
                m_HammerCooldown = 24;
                final_action.m_Hook = false;
                if(current_state.m_IsGrounded > 0.5f && !IsDirectionTowardsHazard(final_action.m_Direction))
                {
                    final_action.m_Jump = true;
                }
            }
        }
        // ---------------------------------------------------------------------
        // SCENARIO 6D: HOOK ACCELERATION & DRAG (Closing distance)
        // ---------------------------------------------------------------------
        else if(EnemyHooked && m_SlingCombosEnabled)
        {
            final_action.m_Tactics = EBlockTactics::OFFENSIVE_DRAG;
            final_action.m_WantedWeapon = 0;
            final_action.m_Hook = true;

            desired_aim_x = enemy_dx + enemy_vx * 0.05f;
            desired_aim_y = enemy_dy + enemy_vy * 0.05f;

            float drag_dir_x = (enemy_dx > 0.0f) ? -1.0f : 1.0f;
            if(current_state.m_HazardDistSelf < 0.8f && current_state.m_HazardDirSelfX != 0.0f)
            {
                drag_dir_x = (current_state.m_HazardDirSelfX < 0.0f) ? 1.0f : -1.0f;
            }
            if(IsDirectionTowardsHazard((int)drag_dir_x))
            {
                drag_dir_x = -drag_dir_x;
            }
            final_action.m_Direction = (drag_dir_x > 0.0f) ? 1 : -1;

            if(enemy_dist < 135.0f && (current_state.m_IsGrounded > 0.5f || current_state.m_CanDoubleJump > 0.5f) && !IsDirectionTowardsHazard(final_action.m_Direction))
            {
                final_action.m_Jump = true;
            }
        }
        // ---------------------------------------------------------------------
        // SCENARIO 6E: NEUTRAL SPACING & LEAD HOOK (DO NOT SPAM HAMMER!)
        // ---------------------------------------------------------------------
        else
        {
            final_action.m_Tactics = EBlockTactics::NEUTRAL_SPACING;
            final_action.m_Fire = false; // NO RANDOM HAMMER SPAM!

            if(m_LeadAimEnabled)
            {
                Math::Vec2 target_rel(enemy_dx, enemy_dy);
                Math::Vec2 target_vel(enemy_vx, enemy_vy);
                Math::Vec2 lead_aim = Math::CalculateLeadAim(target_rel, target_vel, 2800.0f);
                desired_aim_x = lead_aim.x;
                desired_aim_y = lead_aim.y;
            }
            else
            {
                desired_aim_x = enemy_dx;
                desired_aim_y = enemy_dy;
            }

            // Spacing: Maintain 45-65 px distance, never stop directly underneath
            if(std::abs(enemy_dx) < 36.0f)
            {
                final_action.m_Direction = (enemy_dx >= 0.0f) ? 1 : -1;
            }
            else if(enemy_dist > 65.0f)
            {
                final_action.m_Direction = (enemy_dx >= 0.0f) ? 1 : -1;
            }
            else if(enemy_dist < 35.0f)
            {
                final_action.m_Direction = (enemy_dx >= 0.0f) ? -1 : 1;
            }
            else
            {
                final_action.m_Direction = 0;
            }

            // Prevent stepping towards hazard
            if(IsDirectionTowardsHazard(final_action.m_Direction))
            {
                final_action.m_Direction = (current_state.m_HazardDirSelfX > 0.0f) ? -1 : 1;
            }

            // Hook enemy when in medium range (< 380 units) to initiate fly-in or drag
            if(enemy_dist < 380.0f && enemy_dist > 60.0f)
            {
                final_action.m_Hook = true;
            }
        }
    }

    // =========================================================================
    // HOOK RECAST RELIABILITY (Fixes Teeworlds HOOK_RETRACTED lockup)
    // =========================================================================
    if(final_action.m_Hook)
    {
        if(!EnemyHooked)
        {
            m_HookCycleTimer++;
            // If hook has been thrown for 18 ticks without grabbing player/wall, release for 2 ticks to reset
            if(m_HookCycleTimer > 18 && m_HookCycleTimer <= 20)
            {
                final_action.m_Hook = false; // Drop low to cycle HOOK_IDLE
            }
            else if(m_HookCycleTimer > 20)
            {
                m_HookCycleTimer = 0;
            }
        }
        else
        {
            m_HookCycleTimer = 0;
            m_HookHoldTicks++;
        }
    }
    else
    {
        m_HookCycleTimer = 0;
        m_HookHoldTicks = 0;
    }

    // =========================================================================
    // INTELLIGENT TERRAIN JUMPING: Obstacle, Wall, Double Jump, and Gap Clearance
    // =========================================================================
    if(final_action.m_Direction != 0)
    {
        // 1. Ground Jumping for Obstacles / Front Blocks (Ray 0/1 for Right, Ray 8/7 for Left)
        if(current_state.m_IsGrounded > 0.5f)
        {
            if(final_action.m_Direction > 0 && (current_state.m_aWallDist[0] < 0.08f || current_state.m_aWallDist[1] < 0.06f))
            {
                final_action.m_Jump = true; // Jump over right obstacle
            }
            else if(final_action.m_Direction < 0 && (current_state.m_aWallDist[8] < 0.08f || current_state.m_aWallDist[7] < 0.06f))
            {
                final_action.m_Jump = true; // Jump over left obstacle
            }
            else if(EnemyOnHighGround && !DirectlyUnderEnemy)
            {
                final_action.m_Jump = true;
            }
        }
        // 2. Air Double Jump (DJ) for climbing Y-Blocks & Recovery
        else if(current_state.m_CanDoubleJump > 0.5f)
        {
            // Apex / climb double jump when enemy is above
            if(EnemyOnHighGround && self_vy >= -140.0f)
            {
                final_action.m_Jump = true;
            }
            // Emergency double jump if falling towards freeze
            else if(current_state.m_HazardDistSelf < 0.35f && self_vy > 80.0f)
            {
                final_action.m_Jump = true;
            }
        }
    }

    // =========================================================================
    // SMOOTH AIM INTERPOLATION
    // =========================================================================
    if(m_SmoothAim)
    {
        float target_len = std::sqrt(desired_aim_x * desired_aim_x + desired_aim_y * desired_aim_y);
        if(target_len > 1.0f)
        {
            float norm_tx = (desired_aim_x / target_len) * 200.0f;
            float norm_ty = (desired_aim_y / target_len) * 200.0f;

            float alpha_speed = fast_flick_aim ? (m_AimSensitivity * 5.0f) : (m_AimSensitivity * 2.5f);
            float alpha = Math::Clamp(alpha_speed * delta_time, 0.40f, 0.99f);

            m_CurrentAimX = (1.0f - alpha) * m_CurrentAimX + alpha * norm_tx;
            m_CurrentAimY = (1.0f - alpha) * m_CurrentAimY + alpha * norm_ty;

            final_action.m_TargetX = m_CurrentAimX;
            final_action.m_TargetY = m_CurrentAimY;
        }
    }
    else
    {
        final_action.m_TargetX = desired_aim_x;
        final_action.m_TargetY = desired_aim_y;
        m_CurrentAimX = desired_aim_x;
        m_CurrentAimY = desired_aim_y;
    }

    // =========================================================================
    // JUMP CADENCE & DOUBLE JUMP PERMISSION
    // =========================================================================
    if(final_action.m_Jump)
    {
        if(m_JumpCooldown > 0)
        {
            // Allow double jump to pass through if in air and was on cooldown
            if(current_state.m_CanDoubleJump > 0.5f && current_state.m_IsGrounded < 0.5f && m_JumpCooldown <= 2)
            {
                m_JumpCooldown = 4;
            }
            else
            {
                final_action.m_Jump = false;
            }
        }
        else
        {
            m_JumpCooldown = 4; // Fast human click rhythm
        }
    }

    m_LastAction = final_action;
    return final_action;
}

} // namespace AI
