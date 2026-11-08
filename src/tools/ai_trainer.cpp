#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <iomanip>

#include "../ai/ai_engine.h"
#include "../ai/ai_demo_parser.h"

namespace fs = std::filesystem;

void PrintBanner()
{
    std::cout << "=================================================================\n";
    std::cout << "   DDNet (DDRaceNetwork) AI Learning Machine - NeuroTee Engine   \n";
    std::cout << "   Imitation Learning, Demo Trajectory Parsing & Online Learner  \n";
    std::cout << "=================================================================\n\n";
}

void PrintHelp()
{
    std::cout << "Usage: ai_trainer.exe [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --train-demos <dir>       Train on all .demo files in directory\n";
    std::cout << "  --demo <file>             Train on a single .demo file\n";
    std::cout << "  --epochs <count>          Number of training epochs (default: 10)\n";
    std::cout << "  --lr <rate>               Learning rate for Adam optimizer (default: 0.001)\n";
    std::cout << "  --save-model <path>       Save trained model to binary (.bin) or JSON (.json)\n";
    std::cout << "  --load-model <path>       Load existing model before training or evaluation\n";
    std::cout << "  --eval-demos <dir>        Evaluate accuracy metrics on test demos\n";
    std::cout << "  --benchmark               Run speed benchmark of Neural Net and KD-Tree\n";
    std::cout << "  --interactive             Start interactive training console\n";
    std::cout << "  --help                    Show this help message\n\n";
}

void RunBenchmark(AI::AIEngine& engine)
{
    std::cout << "\n[BENCHMARK] Running AI inference and training speed tests...\n";

    AI::AIFeatures state;
    state.m_VelX = 0.45f;
    state.m_VelY = -0.2f;
    state.m_Speed = 0.5f;
    state.m_AngleCos = 0.707f;
    state.m_AngleSin = 0.707f;
    state.m_IsGrounded = 0.0f;
    state.m_CanDoubleJump = 1.0f;
    state.m_HookState = 1.0f;

    // 1. Forward Pass Benchmark
    constexpr int NUM_PREDICTIONS = 200000;
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < NUM_PREDICTIONS; ++i)
    {
        engine.PredictTick(state, 0.02f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end - start).count();
    double preds_per_sec = (double)NUM_PREDICTIONS / total_sec;
    double latency_us = (total_sec / (double)NUM_PREDICTIONS) * 1e6;

    std::cout << "  * Forward Pass Speed: " << std::fixed << std::setprecision(0) << preds_per_sec << " inferences/second\n";
    std::cout << "  * Average Latency:    " << std::fixed << std::setprecision(2) << latency_us << " microseconds / tick (Ultra Low Overhead)\n";

    // 2. Training Step Benchmark
    AI::AIAction act;
    act.m_Direction = 1;
    act.m_Jump = true;
    act.m_Hook = false;
    act.m_TargetX = 180.0f;
    act.m_TargetY = -50.0f;

    AI::AITransition sample;
    sample.m_State = state;
    sample.m_Action = act;
    std::vector<AI::AITransition> batch(64, sample);

    constexpr int NUM_BATCHES = 2000;
    start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < NUM_BATCHES; ++i)
    {
        engine.GetNeuralNet().TrainBatch(batch, 0.001f);
    }

    end = std::chrono::high_resolution_clock::now();
    total_sec = std::chrono::duration<double>(end - start).count();
    double samples_per_sec = (double)(NUM_BATCHES * 64) / total_sec;

    std::cout << "  * Backprop & Adam:    " << std::fixed << std::setprecision(0) << samples_per_sec << " training transitions/second\n\n";
}

int main(int argc, char* argv[])
{
    PrintBanner();

    AI::AIEngine engine;

    std::string train_demos_dir = "";
    std::string single_demo_file = "";
    std::string eval_demos_dir = "";
    std::string save_model_path = "ddnet_ai_model.bin";
    std::string load_model_path = "";
    int epochs = 10;
    float lr = 0.001f;
    bool do_benchmark = false;
    bool interactive = false;

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "--train-demos" && i + 1 < argc)
        {
            train_demos_dir = argv[++i];
        }
        else if(arg == "--demo" && i + 1 < argc)
        {
            single_demo_file = argv[++i];
        }
        else if(arg == "--eval-demos" && i + 1 < argc)
        {
            eval_demos_dir = argv[++i];
        }
        else if(arg == "--save-model" && i + 1 < argc)
        {
            save_model_path = argv[++i];
        }
        else if(arg == "--load-model" && i + 1 < argc)
        {
            load_model_path = argv[++i];
        }
        else if(arg == "--epochs" && i + 1 < argc)
        {
            epochs = std::stoi(argv[++i]);
        }
        else if(arg == "--lr" && i + 1 < argc)
        {
            lr = std::stof(argv[++i]);
        }
        else if(arg == "--benchmark")
        {
            do_benchmark = true;
        }
        else if(arg == "--interactive")
        {
            interactive = true;
        }
        else if(arg == "--help" || arg == "-h")
        {
            PrintHelp();
            return 0;
        }
    }

    if(!load_model_path.empty())
    {
        std::cout << "[MODEL] Loading model from: " << load_model_path << "...\n";
        if(engine.LoadModel(load_model_path))
        {
            std::cout << "[MODEL] Successfully loaded model weights!\n";
        }
        else
        {
            std::cerr << "[MODEL ERROR] Failed to load model file!\n";
        }
    }

    if(do_benchmark)
    {
        RunBenchmark(engine);
    }

    if(!train_demos_dir.empty())
    {
        std::cout << "[TRAIN] Scanning demo directory: " << train_demos_dir << "...\n";
        std::vector<AI::AITransition> dataset;

        size_t count = AI::AIDemoParser::ParseDemoDirectory(train_demos_dir, dataset, [](int cur, int total, const std::string& name) {
            if(cur % 5 == 0 || cur == total)
            {
                std::cout << "\r[PARSING] Processed demo " << cur << " / " << total << " (" << fs::path(name).filename().string() << ")..." << std::flush;
            }
        });

        std::cout << "\n[DATASET] Extracted " << dataset.size() << " valid state-action training samples from " << count << " demos.\n";

        if(!dataset.empty())
        {
            std::cout << "[TRAIN] Training Deep Multi-Head Neural Network (" << epochs << " epochs, lr=" << lr << ")...\n";
            auto start_train = std::chrono::high_resolution_clock::now();

            float final_loss = engine.TrainDataset(dataset, epochs, 64, lr);

            auto end_train = std::chrono::high_resolution_clock::now();
            double train_time = std::chrono::duration<double>(end_train - start_train).count();

            const auto& stats = engine.GetStats();

            std::cout << "\n================ TRAINING RESULTS ================\n";
            std::cout << "  * Training Duration:   " << std::fixed << std::setprecision(2) << train_time << " seconds\n";
            std::cout << "  * Final Loss:          " << std::fixed << std::setprecision(5) << final_loss << "\n";
            std::cout << "  * Direction Accuracy:  " << std::fixed << std::setprecision(1) << (stats.m_DirectionAccuracy * 100.0f) << "%\n";
            std::cout << "  * Jump Accuracy:       " << std::fixed << std::setprecision(1) << (stats.m_JumpAccuracy * 100.0f) << "%\n";
            std::cout << "  * Hook Accuracy:       " << std::fixed << std::setprecision(1) << (stats.m_HookAccuracy * 100.0f) << "%\n";
            std::cout << "  * Aim Error (MSE):     " << std::fixed << std::setprecision(3) << stats.m_AimErrorMSE << "\n";
            std::cout << "==================================================\n\n";

            std::cout << "[SAVE] Saving model to: " << save_model_path << "...\n";
            if(engine.SaveModel(save_model_path))
            {
                std::cout << "[SAVE] Model successfully saved to binary format: " << save_model_path << "\n";
            }

            // Also save JSON companion
            std::string json_path = fs::path(save_model_path).replace_extension(".json").string();
            if(engine.SaveModel(json_path))
            {
                std::cout << "[SAVE] Model config & weights exported to JSON: " << json_path << "\n";
            }
        }
    }
    else if(!single_demo_file.empty())
    {
        std::cout << "[TRAIN] Parsing single demo: " << single_demo_file << "...\n";
        std::vector<AI::AITransition> dataset;
        if(AI::AIDemoParser::ParseDemoFile(single_demo_file, dataset))
        {
            std::cout << "[DATASET] Extracted " << dataset.size() << " samples.\n";
            engine.TrainDataset(dataset, epochs, 64, lr);
            engine.SaveModel(save_model_path);
            std::cout << "[SAVE] Saved trained model to: " << save_model_path << "\n";
        }
        else
        {
            std::cerr << "[ERROR] Could not parse demo file: " << single_demo_file << "\n";
        }
    }
    else if(!interactive && !do_benchmark)
    {
        // Default behavior: check standard DDNet demos folder or run quick self-check
        std::string default_demos = "C:\\Users\\sigma\\AppData\\Roaming\\DDNet\\demos";
        if(fs::exists(default_demos))
        {
            std::cout << "[AUTO] Found standard DDNet demos directory: " << default_demos << "\n";
            std::cout << "[AUTO] Starting automated training on local player demos...\n";

            std::vector<AI::AITransition> dataset;
            size_t count = AI::AIDemoParser::ParseDemoDirectory(default_demos, dataset);
            std::cout << "[DATASET] Loaded " << dataset.size() << " transitions from " << count << " demos.\n";

            if(!dataset.empty())
            {
                engine.TrainDataset(dataset, 10, 64, 0.001f);
                engine.SaveModel("ddnet_ai_model.bin");
                engine.SaveModel("ddnet_ai_model.json");
                std::cout << "[SUCCESS] Initialized and saved pre-trained AI Brain model (ddnet_ai_model.bin)!\n";
            }
        }
        else
        {
            PrintHelp();
        }
    }

    return 0;
}
