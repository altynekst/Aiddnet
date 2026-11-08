#ifndef AI_TYPES_H
#define AI_TYPES_H

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace AI
{

constexpr int FEATURE_DIM = 48;
constexpr int NUM_RAYCASTS = 16;
constexpr float MAX_COORD_SCALE = 1000.0f;
constexpr float MAX_VEL_SCALE = 1500.0f;
constexpr float MAX_DIST_SCALE = 1200.0f;

// High-level Block & PvP Tactical Roles
enum class EBlockTactics
{
    NEUTRAL_SPACING = 0,    // Positioning, spacing, feinting, baiting enemy hook
    OFFENSIVE_DRAG = 1,     // Hooked enemy -> dragging into freeze/death, sling jump
    HAMMER_EXECUTE = 2,     // In close combat -> flick aim and strike with hammer
    DEFENSIVE_RECOVERY = 3, // In danger of freeze -> emergency anchor hook, air jump away
    TEAM_RESCUE = 4,        // Teammate frozen -> unfreeze hammer strike
    FREEZE_SLAM = 5,        // Enemy in shallow freeze -> drag and knock deep into freeze
    FLY_STRIKE = 6          // High velocity fly-in momentum hammer strike
};

// 48-Dimensional State feature vector representing Tee's perception of itself, hazards, and opponents
struct AIFeatures
{
    // 1. Kinematic self state (8)
    float m_VelX = 0.0f;          // Normalized Velocity X [-1, 1]
    float m_VelY = 0.0f;          // Normalized Velocity Y [-1, 1]
    float m_Speed = 0.0f;         // Normalized Speed [0, 1]
    float m_AngleCos = 1.0f;      // Current aim angle cos
    float m_AngleSin = 0.0f;      // Current aim angle sin
    float m_IsGrounded = 0.0f;    // 1.0 if touching ground, else 0.0
    float m_CanDoubleJump = 0.0f; // 1.0 if air jump is available
    float m_IsFrozen = 0.0f;      // 1.0 if frozen, else 0.0

    // 2. Hook state (6)
    float m_HookState = 0.0f;     // 0 = idle, 0.5 = flying, 1.0 = grabbed
    float m_HookDuration = 0.0f;  // Normalized hook duration
    float m_HookDist = 0.0f;      // Distance to hook target
    float m_HookDirX = 0.0f;      // Hook direction vector X
    float m_HookDirY = 0.0f;      // Hook direction vector Y
    float m_HookedPlayer = 0.0f;  // 1.0 if hooked to another tee

    // 3. Environment Proximity Sensors (16 directional raycasts to solid tiles)
    float m_aWallDist[NUM_RAYCASTS] = {0.0f}; // Normalized distance [0 = contact, 1 = open air]

    // 4. Freeze & Hazard Proximity for Self (4)
    float m_HazardDistSelf = 1.0f;// Distance to nearest freeze/death tile (0 = in freeze, 1 = safe)
    float m_HazardDirSelfX = 0.0f;// Normalized direction vector pointing to hazard
    float m_HazardDirSelfY = 0.0f;
    float m_InDangerZone = 0.0f;  // 1.0 if in imminent hazard proximity (< 48 units)

    // 5. Enemy / Opponent relative features for Block (8)
    float m_EnemyRelX = 0.0f;     // Relative delta X to closest opponent
    float m_EnemyRelY = 0.0f;     // Relative delta Y to closest opponent
    float m_EnemyRelVelX = 0.0f;  // Relative velocity X of opponent
    float m_EnemyRelVelY = 0.0f;  // Relative velocity Y of opponent
    float m_EnemyDist = 1.0f;     // Distance to opponent (0..1)
    float m_EnemyInHammerRange = 0.0f; // 1.0 if opponent within hammer hit range (< 64 units)
    float m_EnemyFrozen = 0.0f;   // 1.0 if opponent is frozen
    float m_EnemyHazardDist = 1.0f; // Opponent distance to freeze/death

    // 6. Teammate relative features for Rescue (3)
    float m_TeammateRelX = 0.0f;
    float m_TeammateRelY = 0.0f;
    float m_TeammateFrozen = 0.0f;// 1.0 if partner is frozen (hammer assist)

    // 7. Temporal context (previous actions) (3)
    float m_PrevDirection = 0.0f; // -1, 0, or 1
    float m_PrevJump = 0.0f;
    float m_PrevHook = 0.0f;

    // Convert struct to flat float vector (48 dimensions)
    std::vector<float> ToVector() const
    {
        std::vector<float> vec;
        vec.reserve(FEATURE_DIM);

        // 1. Self kinematics (8)
        vec.push_back(m_VelX);
        vec.push_back(m_VelY);
        vec.push_back(m_Speed);
        vec.push_back(m_AngleCos);
        vec.push_back(m_AngleSin);
        vec.push_back(m_IsGrounded);
        vec.push_back(m_CanDoubleJump);
        vec.push_back(m_IsFrozen);

        // 2. Hook status (6)
        vec.push_back(m_HookState);
        vec.push_back(m_HookDuration);
        vec.push_back(m_HookDist);
        vec.push_back(m_HookDirX);
        vec.push_back(m_HookDirY);
        vec.push_back(m_HookedPlayer);

        // 3. Wall Raycasts (16)
        for(int i = 0; i < NUM_RAYCASTS; ++i)
        {
            vec.push_back(m_aWallDist[i]);
        }

        // 4. Hazard status (4)
        vec.push_back(m_HazardDistSelf);
        vec.push_back(m_HazardDirSelfX);
        vec.push_back(m_HazardDirSelfY);
        vec.push_back(m_InDangerZone);

        // 5. Enemy Block features (8)
        vec.push_back(m_EnemyRelX);
        vec.push_back(m_EnemyRelY);
        vec.push_back(m_EnemyRelVelX);
        vec.push_back(m_EnemyRelVelY);
        vec.push_back(m_EnemyDist);
        vec.push_back(m_EnemyInHammerRange);
        vec.push_back(m_EnemyFrozen);
        vec.push_back(m_EnemyHazardDist);

        // 6. Teammate (3)
        vec.push_back(m_TeammateRelX);
        vec.push_back(m_TeammateRelY);
        vec.push_back(m_TeammateFrozen);

        // 7. Temporal context (3)
        vec.push_back(m_PrevDirection);
        vec.push_back(m_PrevJump);
        vec.push_back(m_PrevHook);

        return vec;
    }

    void FromVector(const std::vector<float>& vec)
    {
        if(vec.size() < FEATURE_DIM) return;
        size_t idx = 0;

        // 1. Self kinematics (8)
        m_VelX = vec[idx++];
        m_VelY = vec[idx++];
        m_Speed = vec[idx++];
        m_AngleCos = vec[idx++];
        m_AngleSin = vec[idx++];
        m_IsGrounded = vec[idx++];
        m_CanDoubleJump = vec[idx++];
        m_IsFrozen = vec[idx++];

        // 2. Hook status (6)
        m_HookState = vec[idx++];
        m_HookDuration = vec[idx++];
        m_HookDist = vec[idx++];
        m_HookDirX = vec[idx++];
        m_HookDirY = vec[idx++];
        m_HookedPlayer = vec[idx++];

        // 3. Wall Raycasts (16)
        for(int i = 0; i < NUM_RAYCASTS; ++i)
            m_aWallDist[i] = vec[idx++];

        // 4. Hazard status (4)
        m_HazardDistSelf = vec[idx++];
        m_HazardDirSelfX = vec[idx++];
        m_HazardDirSelfY = vec[idx++];
        m_InDangerZone = vec[idx++];

        // 5. Enemy Block features (8)
        m_EnemyRelX = vec[idx++];
        m_EnemyRelY = vec[idx++];
        m_EnemyRelVelX = vec[idx++];
        m_EnemyRelVelY = vec[idx++];
        m_EnemyDist = vec[idx++];
        m_EnemyInHammerRange = vec[idx++];
        m_EnemyFrozen = vec[idx++];
        m_EnemyHazardDist = vec[idx++];

        // 6. Teammate (3)
        m_TeammateRelX = vec[idx++];
        m_TeammateRelY = vec[idx++];
        m_TeammateFrozen = vec[idx++];

        // 7. Temporal context (3)
        m_PrevDirection = vec[idx++];
        m_PrevJump = vec[idx++];
        m_PrevHook = vec[idx++];
    }
};

// Target action output predicted by AI or extracted from demo/player
struct AIAction
{
    int m_Direction = 0;       // -1 = Left, 0 = None, 1 = Right
    bool m_Jump = false;       // Trigger jump
    bool m_Hook = false;       // Hold/Trigger hook
    bool m_Fire = false;       // Trigger hammer / weapon
    float m_TargetX = 0.0f;    // Aim cursor vector X
    float m_TargetY = 0.0f;    // Aim cursor vector Y
    int m_WantedWeapon = 0;    // 0 = Hammer, 1 = Gun, 2 = Shotgun, 3 = Grenade, 4 = Laser, 5 = Ninja

    // Tactical State Meta
    EBlockTactics m_Tactics = EBlockTactics::NEUTRAL_SPACING;

    // Probabilities and confidence scores (from Neural Net)
    float m_ProbDirLeft = 0.0f;
    float m_ProbDirNone = 1.0f;
    float m_ProbDirRight = 0.0f;
    float m_ProbJump = 0.0f;
    float m_ProbHook = 0.0f;
    float m_ProbFire = 0.0f;
    float m_Confidence = 1.0f;
};

// Training transition sample
struct AITransition
{
    AIFeatures m_State;
    AIAction m_Action;
    int m_Tick = 0;
    int m_ClientId = 0;
    float m_Weight = 1.0f;     // Priority weight
};

enum class EAIMode
{
    OFF = 0,
    LEARN_LIVE = 1,            // Continuously watch other players and learn online
    AUTOPILOT = 2,             // AI controls the local character / dummy
    ASSIST = 3                 // AI assists player (unfreeze hammer, catch hook, jump sync)
};

struct AITrainingStats
{
    uint64_t m_TotalSamplesRecorded = 0;
    uint64_t m_TotalTrainingSteps = 0;
    float m_RecentLoss = 0.0f;
    float m_DirectionAccuracy = 0.0f;
    float m_JumpAccuracy = 0.0f;
    float m_HookAccuracy = 0.0f;
    float m_AimErrorMSE = 0.0f;
    int m_TotalDemosParsed = 0;
    double m_LastInferenceMicroseconds = 0.0;
};

} // namespace AI

#endif // AI_TYPES_H
