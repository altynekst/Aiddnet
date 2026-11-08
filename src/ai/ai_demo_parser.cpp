#include "ai_demo_parser.h"
#include "ai_math.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <map>

namespace AI
{

namespace fs = std::filesystem;

struct RawDemoHeader
{
    char m_aMarker[7];
    unsigned char m_Version;
    char m_aNetVersion[64];
    char m_aMapName[64];
    unsigned char m_aMapSize[4];
    unsigned char m_aMapCrc[4];
    char m_aType[8];
    char m_aLength[4];
    char m_aTimestamp[20];
};

struct PlayerTickRecord
{
    int m_Tick = 0;
    float m_PosX = 0.0f;
    float m_PosY = 0.0f;
    float m_VelX = 0.0f;
    float m_VelY = 0.0f;
    int m_Angle = 0;
    int m_Direction = 0;
    int m_Jumped = 0;
    int m_HookedPlayer = -1;
    int m_HookState = 0;
    int m_HookTick = 0;
    float m_HookX = 0.0f;
    float m_HookY = 0.0f;
    int m_Weapon = 0;
    bool m_IsFrozen = false;
};

bool AIDemoParser::ParseDemoFile(const std::string& filepath, std::vector<AITransition>& out_transitions)
{
    std::ifstream file(filepath, std::ios::binary);
    if(!file.is_open()) return false;

    // Check size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if(file_size < 128) return false;

    // Read header
    RawDemoHeader header;
    file.read((char*)&header, sizeof(header));
    if(!file.good()) return false;

    // Heuristic stream parser to extract player state sequences across ticks
    std::map<int, std::vector<PlayerTickRecord>> player_trajectories;

    std::vector<unsigned char> buffer(file_size);
    file.seekg(0, std::ios::beg);
    file.read((char*)buffer.data(), file_size);

    // Scan for character snapshot patterns in uncompressed/chunked streams
    int current_tick = 0;
    for(size_t i = 128; i + 32 < file_size; i += 4)
    {
        // Snapshot / Tick marker heuristics
        int32_t val0 = *(int32_t*)&buffer[i];
        int32_t val1 = *(int32_t*)&buffer[i + 4];
        int32_t val2 = *(int32_t*)&buffer[i + 8];
        int32_t val3 = *(int32_t*)&buffer[i + 12];
        int32_t val4 = *(int32_t*)&buffer[i + 16];
        int32_t val5 = *(int32_t*)&buffer[i + 20];
        int32_t val6 = *(int32_t*)&buffer[i + 24];

        // DDNet character snapshot: X, Y, VelX, VelY, Angle, Direction, Jumped, HookedPlayer
        // X and Y are coordinate units, VelX/VelY typically in range [-4000, 4000]
        if(std::abs(val0) < 500000 && std::abs(val1) < 500000 &&
           std::abs(val2) < 6000 && std::abs(val3) < 6000 &&
           val4 >= -1000 && val4 <= 1000 &&
           (val5 >= -1 && val5 <= 1) &&
           (val6 >= 0 && val6 <= 3))
        {
            PlayerTickRecord rec;
            rec.m_Tick = current_tick++;
            rec.m_PosX = (float)val0;
            rec.m_PosY = (float)val1;
            rec.m_VelX = (float)val2;
            rec.m_VelY = (float)val3;
            rec.m_Angle = val4;
            rec.m_Direction = val5;
            rec.m_Jumped = val6;

            if(i + 48 < file_size)
            {
                rec.m_HookedPlayer = *(int32_t*)&buffer[i + 28];
                rec.m_HookState = *(int32_t*)&buffer[i + 32];
                rec.m_HookTick = *(int32_t*)&buffer[i + 36];
                rec.m_HookX = (float)*(int32_t*)&buffer[i + 40];
                rec.m_HookY = (float)*(int32_t*)&buffer[i + 44];
            }

            player_trajectories[0].push_back(rec);
            i += 44; // Advance past character block
        }
    }

    // Convert raw records into continuous training transitions
    for(auto& [client_id, records] : player_trajectories)
    {
        if(records.size() < 10) continue;

        for(size_t i = 1; i + 1 < records.size(); ++i)
        {
            const auto& prev = records[i - 1];
            const auto& curr = records[i];
            const auto& next = records[i + 1];

            AIFeatures feat;
            feat.m_VelX = Math::Clamp(curr.m_VelX / MAX_VEL_SCALE, -1.0f, 1.0f);
            feat.m_VelY = Math::Clamp(curr.m_VelY / MAX_VEL_SCALE, -1.0f, 1.0f);
            feat.m_Speed = Math::Clamp(std::sqrt(curr.m_VelX * curr.m_VelX + curr.m_VelY * curr.m_VelY) / MAX_VEL_SCALE, 0.0f, 1.0f);

            float rad = (float)curr.m_Angle / 256.0f;
            feat.m_AngleCos = std::cos(rad);
            feat.m_AngleSin = std::sin(rad);

            feat.m_IsGrounded = (std::abs(curr.m_VelY) < 1.0f && std::abs(prev.m_VelY) < 1.0f) ? 1.0f : 0.0f;
            feat.m_CanDoubleJump = (curr.m_Jumped == 0) ? 1.0f : 0.0f;
            feat.m_IsFrozen = curr.m_IsFrozen ? 1.0f : 0.0f;

            feat.m_HookState = (curr.m_HookState > 0) ? ((curr.m_HookState == 2) ? 1.0f : 0.5f) : 0.0f;
            feat.m_HookDuration = Math::Clamp((float)curr.m_HookTick / 250.0f, 0.0f, 1.0f);
            float hook_dx = curr.m_HookX - curr.m_PosX;
            float hook_dy = curr.m_HookY - curr.m_PosY;
            float hook_dist = std::sqrt(hook_dx * hook_dx + hook_dy * hook_dy);
            feat.m_HookDist = Math::Clamp(hook_dist / MAX_DIST_SCALE, 0.0f, 1.0f);
            if(hook_dist > 1.0f)
            {
                feat.m_HookDirX = hook_dx / hook_dist;
                feat.m_HookDirY = hook_dy / hook_dist;
            }
            feat.m_HookedPlayer = (curr.m_HookedPlayer >= 0) ? 1.0f : 0.0f;

            // Approximated raycasts based on velocity changes and position
            for(int r = 0; r < NUM_RAYCASTS; ++r)
            {
                float ray_angle = (float)r * (2.0f * 3.14159265f / (float)NUM_RAYCASTS);
                feat.m_aWallDist[r] = 1.0f; // Default open
                if(r == 4 && feat.m_IsGrounded > 0.5f) feat.m_aWallDist[r] = 0.05f; // Down ray near ground
            }

            feat.m_PrevDirection = (float)prev.m_Direction;
            feat.m_PrevJump = (prev.m_Jumped > 0) ? 1.0f : 0.0f;
            feat.m_PrevHook = (prev.m_HookState > 0) ? 1.0f : 0.0f;

            // Target ground truth actions executed at this tick
            AIAction act;
            act.m_Direction = curr.m_Direction;
            if(act.m_Direction == 0 && std::abs(next.m_VelX - curr.m_VelX) > 20.0f)
            {
                act.m_Direction = (next.m_VelX > curr.m_VelX) ? 1 : -1;
            }

            act.m_Jump = (next.m_Jumped > curr.m_Jumped) || (curr.m_VelY < -250.0f && prev.m_VelY >= -50.0f);
            act.m_Hook = (curr.m_HookState > 0) || (next.m_HookState > 0);
            act.m_TargetX = feat.m_AngleCos * 200.0f;
            act.m_TargetY = feat.m_AngleSin * 200.0f;
            act.m_Fire = (curr.m_Weapon != prev.m_Weapon);

            AITransition trans;
            trans.m_State = feat;
            trans.m_Action = act;
            trans.m_Tick = curr.m_Tick;
            trans.m_ClientId = client_id;
            trans.m_Weight = 1.0f;

            out_transitions.push_back(trans);
        }
    }

    return !out_transitions.empty();
}

size_t AIDemoParser::ParseDemoDirectory(const std::string& directory_path, std::vector<AITransition>& out_transitions, ProgressCallback progress_fn)
{
    if(!fs::exists(directory_path)) return 0;

    std::vector<std::string> demo_files;
    for(const auto& entry : fs::recursive_directory_iterator(directory_path, fs::directory_options::skip_permission_denied))
    {
        if(entry.is_regular_file())
        {
            std::string ext = entry.path().extension().string();
            if(ext == ".demo" || ext == ".dmd")
            {
                demo_files.push_back(entry.path().string());
            }
        }
    }

    size_t total_files = demo_files.size();
    size_t parsed_count = 0;

    for(size_t i = 0; i < total_files; ++i)
    {
        const auto& path = demo_files[i];
        if(progress_fn)
        {
            progress_fn((int)(i + 1), (int)total_files, path);
        }

        std::vector<AITransition> file_trans;
        if(ParseDemoFile(path, file_trans))
        {
            out_transitions.insert(out_transitions.end(), file_trans.begin(), file_trans.end());
            parsed_count++;
        }
    }

    return parsed_count;
}

} // namespace AI
