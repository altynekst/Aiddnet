#ifndef AI_DEMO_PARSER_H
#define AI_DEMO_PARSER_H

#include "ai_types.h"
#include <vector>
#include <string>
#include <functional>

namespace AI
{

class AIDemoParser
{
public:
    using ProgressCallback = std::function<void(int current, int total, const std::string& filename)>;

    AIDemoParser() = default;

    // Parse a single .demo or .dmd file and extract training transitions
    static bool ParseDemoFile(const std::string& filepath, std::vector<AITransition>& out_transitions);

    // Recursively find and parse all .demo files in a directory
    static size_t ParseDemoDirectory(const std::string& directory_path, std::vector<AITransition>& out_transitions, ProgressCallback progress_fn = nullptr);

    // Reconstruct input actions from raw character trajectories
    static void ProcessTrajectory(const std::vector<AIFeatures>& states, const std::vector<int>& ticks, std::vector<AITransition>& out_transitions);
};

} // namespace AI

#endif // AI_DEMO_PARSER_H
