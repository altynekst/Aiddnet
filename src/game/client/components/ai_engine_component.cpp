#include "ai_engine_component.h"
#include <game/client/gameclient.h>
#include <game/client/components/controls.h>
#include <game/client/prediction/entities/character.h>
#include <game/collision.h>
#include <game/mapitems.h>
#include <engine/textrender.h>
#include <engine/graphics.h>
#include <engine/console.h>
#include <engine/shared/config.h>
#include <base/math.h>
#include <base/str.h>
#include <base/time.h>
#include <base/types.h>

#include <cmath>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

static bool IsHazardTile(const CCollision *pCollision, float x, float y)
{
	if(!pCollision) return false;
	int idx = pCollision->GetPureMapIndex(x, y);
	if(idx < 0) return false;

	int tile = pCollision->GetTileIndex(idx);
	int front = pCollision->GetFrontTileIndex(idx);

	if(tile == TILE_FREEZE || tile == TILE_DFREEZE || tile == TILE_LFREEZE || tile == TILE_DEATH ||
	   front == TILE_FREEZE || front == TILE_DFREEZE || front == TILE_LFREEZE || front == TILE_DEATH)
		return true;

	if(pCollision->IsEvilTeleport(idx) || pCollision->IsCheckEvilTeleport(idx))
		return true;

	return false;
}

CAIEngineComponent::CAIEngineComponent() :
	m_pEngine(std::make_unique<AI::AIEngine>())
{
}

CAIEngineComponent::~CAIEngineComponent() = default;

std::string CAIEngineComponent::GetDefaultModelPath() const
{
	if(fs::exists("ddnet_ai_model.bin"))
		return "ddnet_ai_model.bin";
	if(fs::exists("models/ddnet_ai_model.bin"))
		return "models/ddnet_ai_model.bin";
	if(fs::exists("ddnet_ai_model.json"))
		return "ddnet_ai_model.json";
	return "ddnet_ai_model.bin";
}

void CAIEngineComponent::SetStatus(const char *pMsg)
{
	str_copy(m_aStatusMsg, pMsg, sizeof(m_aStatusMsg));
	m_StatusMsgTime = time_get();
}

void CAIEngineComponent::OnInit()
{
	m_pEngine->Init();

	// Explicitly disable automatic demo and race recording to prevent background disk writes
	g_Config.m_ClAutoDemoRecord = 0;
	g_Config.m_ClAutoRaceRecord = 0;
	g_Config.m_SvAutoDemoRecord = 0;

	// Attempt auto-loading pretrained model
	std::string path = GetDefaultModelPath();
	if(fs::exists(path))
	{
		if(m_pEngine->LoadModel(path))
		{
			m_ModelLoaded = true;
			SetStatus("Нейросеть NeuroBlock AI успешно загружена");
		}
	}
}

void CAIEngineComponent::OnReset()
{
	if(m_pEngine)
	{
		m_pEngine->GetController().Reset();
	}
	m_WasFrozen = false;
	m_OnlineTrainCounter = 0;
	m_FireCounter = 0;
}

void CAIEngineComponent::OnConsoleInit()
{
	Console()->Register("ai_train_demos", "", CFGFLAG_CLIENT, ConTrainDemos, this, "Обучить AI на всех локальных демо-записях");
	Console()->Register("ai_save_model", "?s[filename]", CFGFLAG_CLIENT, ConSaveModel, this, "Сохранить веса нейросети AI");
	Console()->Register("ai_load_model", "?s[filename]", CFGFLAG_CLIENT, ConLoadModel, this, "Загрузить веса нейросети AI");
	Console()->Register("ai_reset_model", "", CFGFLAG_CLIENT, ConResetModel, this, "Сбросить веса нейросети AI");
	Console()->Register("ai_status", "", CFGFLAG_CLIENT, ConStatus, this, "Показать диагностику и статистику AI");
}

void CAIEngineComponent::ConTrainDemos(IConsole::IResult *pResult, void *pUserData)
{
	auto *pSelf = static_cast<CAIEngineComponent *>(pUserData);
	pSelf->TrainOnDemos();
}

void CAIEngineComponent::ConSaveModel(IConsole::IResult *pResult, void *pUserData)
{
	auto *pSelf = static_cast<CAIEngineComponent *>(pUserData);
	const char *pFile = pResult->NumArguments() > 0 ? pResult->GetString(0) : nullptr;
	pSelf->SaveModel(pFile);
}

void CAIEngineComponent::ConLoadModel(IConsole::IResult *pResult, void *pUserData)
{
	auto *pSelf = static_cast<CAIEngineComponent *>(pUserData);
	const char *pFile = pResult->NumArguments() > 0 ? pResult->GetString(0) : nullptr;
	pSelf->LoadModel(pFile);
}

void CAIEngineComponent::ConResetModel(IConsole::IResult *pResult, void *pUserData)
{
	auto *pSelf = static_cast<CAIEngineComponent *>(pUserData);
	pSelf->ResetModel();
}

void CAIEngineComponent::ConStatus(IConsole::IResult *pResult, void *pUserData)
{
	auto *pSelf = static_cast<CAIEngineComponent *>(pUserData);
	const auto &stats = pSelf->m_pEngine->GetStats();
	char aBuf[256];
	str_format(aBuf, sizeof(aBuf), "AI Статус: Демок=%d Семплов=%llu Движение=%.1f%% Прыжки=%.1f%% Хук=%.1f%% Ошибка=%.4f",
		stats.m_TotalDemosParsed,
		(unsigned long long)stats.m_TotalSamplesRecorded,
		stats.m_DirectionAccuracy * 100.0f,
		stats.m_JumpAccuracy * 100.0f,
		stats.m_HookAccuracy * 100.0f,
		stats.m_RecentLoss);
	pSelf->Console()->Print(IConsole::OUTPUT_LEVEL_STANDARD, "ai", aBuf);
}

void CAIEngineComponent::TrainOnDemos()
{
	SetStatus("Поиск файлов демок...");
	std::vector<std::string> search_dirs = {
		"demos",
		"data/demos"
	};

	const char *pAppdata = getenv("APPDATA");
	if(pAppdata)
	{
		std::string appdata_demos = std::string(pAppdata) + "/DDNet/demos";
		search_dirs.push_back(appdata_demos);
	}

	size_t total_trained = 0;
	for(const auto &dir : search_dirs)
	{
		if(fs::exists(dir))
		{
			size_t n = m_pEngine->TrainOnDemoDirectory(dir, 15, 0.001f, nullptr);
			total_trained += n;
		}
	}

	if(total_trained > 0)
	{
		m_ModelLoaded = true;
		m_pEngine->SaveModel("ddnet_ai_model.bin");
		m_pEngine->SaveModel("ddnet_ai_model.json");
		char aBuf[128];
		str_format(aBuf, sizeof(aBuf), "Обучение завершено! (Обработано %zu демок)", total_trained);
		SetStatus(aBuf);
	}
	else
	{
		SetStatus("Файлы .demo не найдены в папке демок");
	}
}

void CAIEngineComponent::SaveModel(const char *pFilename)
{
	std::string path = (pFilename && pFilename[0]) ? pFilename : "ddnet_ai_model.bin";
	if(m_pEngine->SaveModel(path))
	{
		char aBuf[128];
		str_format(aBuf, sizeof(aBuf), "Модель сохранена в %s", path.c_str());
		SetStatus(aBuf);
	}
	else
	{
		SetStatus("Не удалось сохранить модель");
	}
}

void CAIEngineComponent::LoadModel(const char *pFilename)
{
	std::string path = (pFilename && pFilename[0]) ? pFilename : GetDefaultModelPath();
	if(m_pEngine->LoadModel(path))
	{
		m_ModelLoaded = true;
		char aBuf[128];
		str_format(aBuf, sizeof(aBuf), "Модель загружена из %s", path.c_str());
		SetStatus(aBuf);
	}
	else
	{
		SetStatus("Не удалось загрузить файл модели");
	}
}

void CAIEngineComponent::ResetModel()
{
	m_pEngine->GetNeuralNet().Init(AI::FEATURE_DIM, 128, 128, 64);
	m_pEngine->GetKDTree().Clear();
	m_pEngine->GetController().Reset();
	m_ModelLoaded = false;
	SetStatus("Веса нейросети сброшены");
}

void CAIEngineComponent::ExtractFeatures(int DummyIndex, AI::AIFeatures &out_features) const
{
	out_features = AI::AIFeatures();

	int LocalId = GameClient()->m_Snap.m_LocalClientId;
	if(LocalId < 0 || LocalId >= MAX_CLIENTS || !GameClient()->m_Snap.m_aCharacters[LocalId].m_Active)
		return;

	// Helper to determine fine-grained freeze status:
	// 0 = Unfrozen (active threat)
	// 1 = Shallow/Temporary freeze (thawable / in freeze tile -> target to finish off!)
	// 2 = Deep freeze (incapacitated)
	auto GetPlayerFreezeState = [&](int ClientId) -> int {
		if(ClientId < 0 || ClientId >= MAX_CLIENTS)
			return 0;
		const auto &ClientData = GameClient()->m_aClients[ClientId];
		if(ClientData.m_DeepFrozen || ClientData.m_FreezeEnd == -1)
			return 2;
		if(ClientData.m_FreezeEnd > 0 || ClientData.m_LiveFrozen)
			return 1;
		CCharacter *pChar = GameClient()->m_RegularPredictedWorld.GetCharacterById(ClientId);
		if(pChar && (pChar->Core()->m_IsInFreeze || pChar->m_FreezeTime > 0))
			return 1;
		vec2 check_pos = ClientData.m_RegularPredicted.m_Pos;
		if(check_pos.x == 0.0f && check_pos.y == 0.0f && GameClient()->m_Snap.m_aCharacters[ClientId].m_Active)
		{
			check_pos = vec2(GameClient()->m_Snap.m_aCharacters[ClientId].m_Cur.m_X, GameClient()->m_Snap.m_aCharacters[ClientId].m_Cur.m_Y);
		}
		if(check_pos.x != 0.0f || check_pos.y != 0.0f)
		{
			if(IsHazardTile(Collision(), check_pos.x, check_pos.y) ||
			   IsHazardTile(Collision(), check_pos.x - 14.0f, check_pos.y) ||
			   IsHazardTile(Collision(), check_pos.x + 14.0f, check_pos.y) ||
			   IsHazardTile(Collision(), check_pos.x, check_pos.y + 16.0f) ||
			   IsHazardTile(Collision(), check_pos.x, check_pos.y - 14.0f))
				return 1;
		}
		return 0;
	};

	vec2 Pos = GameClient()->m_aClients[LocalId].m_RegularPredicted.m_Pos;
	vec2 Vel = GameClient()->m_aClients[LocalId].m_RegularPredicted.m_Vel;
	if(Pos.x == 0.0f && Pos.y == 0.0f && GameClient()->m_Snap.m_aCharacters[LocalId].m_Active)
	{
		Pos = vec2(GameClient()->m_Snap.m_aCharacters[LocalId].m_Cur.m_X, GameClient()->m_Snap.m_aCharacters[LocalId].m_Cur.m_Y);
	}

	// 1. Kinematics (8)
	out_features.m_VelX = std::clamp(Vel.x / AI::MAX_VEL_SCALE, -1.0f, 1.0f);
	out_features.m_VelY = std::clamp(Vel.y / AI::MAX_VEL_SCALE, -1.0f, 1.0f);
	out_features.m_Speed = std::clamp(length(Vel) / AI::MAX_VEL_SCALE, 0.0f, 1.0f);

	const float TargetX = (float)GameClient()->m_Controls.m_aInputData[DummyIndex].m_TargetX;
	const float TargetY = (float)GameClient()->m_Controls.m_aInputData[DummyIndex].m_TargetY;
	const float Angle = std::atan2(TargetY, TargetX);
	out_features.m_AngleCos = std::cos(Angle);
	out_features.m_AngleSin = std::sin(Angle);

	// Ground check
	const bool Grounded = Collision()->CheckPoint(Pos.x - 12.0f, Pos.y + 16.0f) ||
	                      Collision()->CheckPoint(Pos.x + 12.0f, Pos.y + 16.0f);
	out_features.m_IsGrounded = Grounded ? 1.0f : 0.0f;

	if(GameClient()->m_Snap.m_pLocalCharacter)
	{
		out_features.m_CanDoubleJump = (GameClient()->m_Snap.m_pLocalCharacter->m_Jumped < 2) ? 1.0f : 0.0f;
		out_features.m_HookState = (float)GameClient()->m_Snap.m_pLocalCharacter->m_HookState / 2.0f;
		out_features.m_HookedPlayer = (GameClient()->m_Snap.m_pLocalCharacter->m_HookedPlayer >= 0) ? 1.0f : 0.0f;
	}
	else
	{
		out_features.m_CanDoubleJump = 1.0f;
	}

	out_features.m_IsFrozen = (GetPlayerFreezeState(LocalId) > 0) ? 1.0f : 0.0f;

	// 2. Hook status (6)
	if(GameClient()->m_Snap.m_pLocalCharacter)
	{
		out_features.m_HookDuration = 0.5f;
		out_features.m_HookDist = 0.5f;
		out_features.m_HookDirX = out_features.m_AngleCos;
		out_features.m_HookDirY = out_features.m_AngleSin;
	}

	// 3. 16 Raycasts around Tee (Solid Wall Distance)
	constexpr int NUM_RAYS = AI::NUM_RAYCASTS;
	for(int i = 0; i < NUM_RAYS; i++)
	{
		const float RayAngle = (float)i * (2.0f * pi / (float)NUM_RAYS);
		const vec2 RayDir = direction(RayAngle);
		vec2 HitPos, HitBefore;
		int Hit = Collision()->IntersectLine(Pos, Pos + RayDir * AI::MAX_DIST_SCALE, &HitPos, &HitBefore);
		if(Hit)
		{
			float Dist = length(HitPos - Pos);
			out_features.m_aWallDist[i] = std::clamp(Dist / AI::MAX_DIST_SCALE, 0.0f, 1.0f);
		}
		else
		{
			out_features.m_aWallDist[i] = 1.0f;
		}
	}

	// 4. Hazard & Freeze Matrix Radar + Real Trajectory Scanner (4)
	float ClosestHazardDist = 1e9f;
	vec2 ClosestHazardDir = vec2(0.0f, 0.0f);

	// A) Matrix scan: 15x15 tiles around player (~240 px radius)
	for(int dy = -7; dy <= 7; ++dy)
	{
		for(int dx = -7; dx <= 7; ++dx)
		{
			vec2 CheckP = Pos + vec2((float)dx * 32.0f, (float)dy * 32.0f);
			if(IsHazardTile(Collision(), CheckP.x, CheckP.y))
			{
				float d = length(CheckP - Pos);
				if(d < ClosestHazardDist)
				{
					ClosestHazardDist = d;
					ClosestHazardDir = CheckP - Pos;
				}
			}
		}
	}

	// B) Forward Trajectory Lookahead Scanner (detects if fast inertia will throw us directly into freeze)
	bool TrajectoryInDanger = false;
	if(length(Vel) > 150.0f)
	{
		const float SimTimes[] = {0.08f, 0.16f, 0.25f, 0.4f};
		for(float dt : SimTimes)
		{
			vec2 SimP = Pos + Vel * dt;
			if(IsHazardTile(Collision(), SimP.x, SimP.y))
			{
				TrajectoryInDanger = true;
				float sim_d = length(SimP - Pos);
				if(sim_d < ClosestHazardDist)
				{
					ClosestHazardDist = sim_d;
					ClosestHazardDir = Vel;
				}
				break;
			}
		}
	}

	if(ClosestHazardDist < 1e8f)
	{
		out_features.m_HazardDistSelf = std::clamp(ClosestHazardDist / 400.0f, 0.0f, 1.0f);
		float len = length(ClosestHazardDir);
		if(len > 0.001f)
		{
			out_features.m_HazardDirSelfX = ClosestHazardDir.x / len;
			out_features.m_HazardDirSelfY = ClosestHazardDir.y / len;
		}
		// Danger zone triggers on immediate proximity (< 52 units) or direct fast collision trajectory
		out_features.m_InDangerZone = (ClosestHazardDist < 52.0f || (TrajectoryInDanger && ClosestHazardDist < 100.0f)) ? 1.0f : 0.0f;
	}
	else
	{
		out_features.m_HazardDistSelf = 1.0f;
		out_features.m_InDangerZone = 0.0f;
	}

	auto IsTeammate = [&](int ClientId) -> bool {
		if(ClientId < 0 || ClientId >= MAX_CLIENTS || ClientId == LocalId)
			return false;
		// In Block Mode, all other players are strictly ENEMIES
		if(g_Config.m_AiBlockMode)
			return false;
		if(GameClient()->m_Teams.Team(LocalId) > 0 && GameClient()->m_Teams.SameTeam(LocalId, ClientId))
			return true;
		if(GameClient()->m_Snap.m_pGameInfoObj && (GameClient()->m_Snap.m_pGameInfoObj->m_GameFlags & GAMEFLAG_TEAMS))
		{
			return GameClient()->m_aClients[LocalId].m_Team == GameClient()->m_aClients[ClientId].m_Team &&
			       GameClient()->m_aClients[LocalId].m_Team != 0;
		}
		return false;
	};

	// 5. Intelligent Multi-Tier Opponent Target Selection (8)
	float BestActiveDist = 1e9f;
	int BestActiveId = -1;

	float BestShallowDist = 1e9f;
	int BestShallowId = -1;

	float BestDeepDist = 1e9f;
	int BestDeepId = -1;

	// 6. Teammate for Rescue (3)
	float BestTeammateDist = 1e9f;
	int BestTeammateId = -1;

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(i == LocalId || !GameClient()->m_Snap.m_aCharacters[i].m_Active)
			continue;

		vec2 OtherPos = GameClient()->m_aClients[i].m_RegularPredicted.m_Pos;
		if(OtherPos.x == 0.0f && OtherPos.y == 0.0f)
		{
			OtherPos = vec2(GameClient()->m_Snap.m_aCharacters[i].m_Cur.m_X, GameClient()->m_Snap.m_aCharacters[i].m_Cur.m_Y);
		}

		const float D = length(OtherPos - Pos);
		int FreezeState = GetPlayerFreezeState(i);
		bool IsTeam = IsTeammate(i);

		if(IsTeam)
		{
			if(FreezeState > 0 && D < BestTeammateDist)
			{
				BestTeammateDist = D;
				BestTeammateId = i;
			}
		}
		else
		{
			// Strictly an enemy
			if(FreezeState == 0)
			{
				if(D < BestActiveDist)
				{
					BestActiveDist = D;
					BestActiveId = i;
				}
			}
			else if(FreezeState == 1)
			{
				if(D < BestShallowDist)
				{
					BestShallowDist = D;
					BestShallowId = i;
				}
			}
			else // Deep frozen
			{
				if(D < BestDeepDist)
				{
					BestDeepDist = D;
					BestDeepId = i;
				}
			}
		}
	}

	// 5. Intelligent Multi-Tier Opponent Target Selection with Sticky Locking
	int BestEnemyId = -1;
	float BestEnemyDist = 1e9f;
	int BestEnemyFreezeState = 0;

	// Check if current locked enemy is still valid & active
	if(m_LockedEnemyId >= 0 && m_LockedEnemyId < MAX_CLIENTS &&
	   m_LockedEnemyId != LocalId && GameClient()->m_Snap.m_aCharacters[m_LockedEnemyId].m_Active)
	{
		vec2 LockedPos = GameClient()->m_aClients[m_LockedEnemyId].m_RegularPredicted.m_Pos;
		if(LockedPos.x == 0.0f && LockedPos.y == 0.0f)
			LockedPos = vec2(GameClient()->m_Snap.m_aCharacters[m_LockedEnemyId].m_Cur.m_X, GameClient()->m_Snap.m_aCharacters[m_LockedEnemyId].m_Cur.m_Y);

		float LockedDist = length(LockedPos - Pos);
		int LockedFreeze = GetPlayerFreezeState(m_LockedEnemyId);

		// If locked target is within 750px and not deep frozen (or if no other closer active exists), stick to them!
		if(LockedDist < 750.0f && (LockedFreeze < 2 || BestActiveId < 0))
		{
			// Give locked target a strong stickiness advantage (160px hysteresis bias)
			float BiasedDist = LockedDist - 160.0f;
			if(BestActiveId < 0 || BestActiveId == m_LockedEnemyId || BestActiveDist > BiasedDist)
			{
				BestEnemyId = m_LockedEnemyId;
				BestEnemyDist = LockedDist;
				BestEnemyFreezeState = LockedFreeze;
			}
		}
	}

	// If no valid locked target or new target is significantly closer:
	if(BestEnemyId < 0)
	{
		// Prioritize nearby shallow frozen enemy (e.g. on Y-block or platform) if they are close (< 280px)
		if(BestShallowId >= 0 && BestShallowDist < 280.0f && (BestActiveId < 0 || BestActiveDist > BestShallowDist + 120.0f))
		{
			BestEnemyId = BestShallowId;
			BestEnemyDist = BestShallowDist;
			BestEnemyFreezeState = 1;
		}
		else if(BestActiveId >= 0)
		{
			BestEnemyId = BestActiveId;
			BestEnemyDist = BestActiveDist;
			BestEnemyFreezeState = 0;
		}
		else if(BestShallowId >= 0)
		{
			BestEnemyId = BestShallowId;
			BestEnemyDist = BestShallowDist;
			BestEnemyFreezeState = 1;
		}
		else if(BestDeepId >= 0)
		{
			BestEnemyId = BestDeepId;
			BestEnemyDist = BestDeepDist;
			BestEnemyFreezeState = 2;
		}
	}

	m_LockedEnemyId = BestEnemyId;

	if(BestEnemyId >= 0)
	{
		vec2 EnemyPos = GameClient()->m_aClients[BestEnemyId].m_RegularPredicted.m_Pos;
		vec2 EnemyVel = GameClient()->m_aClients[BestEnemyId].m_RegularPredicted.m_Vel;
		if(EnemyPos.x == 0.0f && EnemyPos.y == 0.0f)
		{
			EnemyPos = vec2(GameClient()->m_Snap.m_aCharacters[BestEnemyId].m_Cur.m_X, GameClient()->m_Snap.m_aCharacters[BestEnemyId].m_Cur.m_Y);
		}

		out_features.m_EnemyRelX = std::clamp((EnemyPos.x - Pos.x) / AI::MAX_DIST_SCALE, -1.0f, 1.0f);
		out_features.m_EnemyRelY = std::clamp((EnemyPos.y - Pos.y) / AI::MAX_DIST_SCALE, -1.0f, 1.0f);
		out_features.m_EnemyRelVelX = std::clamp((EnemyVel.x - Vel.x) / AI::MAX_VEL_SCALE, -1.0f, 1.0f);
		out_features.m_EnemyRelVelY = std::clamp((EnemyVel.y - Vel.y) / AI::MAX_VEL_SCALE, -1.0f, 1.0f);
		out_features.m_EnemyDist = std::clamp(BestEnemyDist / AI::MAX_DIST_SCALE, 0.0f, 1.0f);
		out_features.m_EnemyInHammerRange = (BestEnemyDist <= 84.0f) ? 1.0f : 0.0f;

		// 0.0f = Unfrozen, 0.5f = Shallow Freeze (Finisher target), 1.0f = Deep Freeze
		out_features.m_EnemyFrozen = (BestEnemyFreezeState == 1) ? 0.5f : ((BestEnemyFreezeState == 2) ? 1.0f : 0.0f);

		// Compute enemy distance to nearest hazard
		float EnemyHazardDist = 1e9f;
		for(int dy = -6; dy <= 6; ++dy)
		{
			for(int dx = -6; dx <= 6; ++dx)
			{
				vec2 CheckP = EnemyPos + vec2((float)dx * 32.0f, (float)dy * 32.0f);
				if(IsHazardTile(Collision(), CheckP.x, CheckP.y))
				{
					float d = length(CheckP - EnemyPos);
					if(d < EnemyHazardDist)
					{
						EnemyHazardDist = d;
					}
				}
			}
		}
		out_features.m_EnemyHazardDist = (EnemyHazardDist < 1e8f) ? std::clamp(EnemyHazardDist / 400.0f, 0.0f, 1.0f) : 1.0f;
	}
	else
	{
		out_features.m_EnemyDist = 1.0f;
	}

	if(BestTeammateId >= 0)
	{
		const vec2 TeamPos = GameClient()->m_aClients[BestTeammateId].m_RegularPredicted.m_Pos;
		out_features.m_TeammateRelX = std::clamp((TeamPos.x - Pos.x) / AI::MAX_DIST_SCALE, -1.0f, 1.0f);
		out_features.m_TeammateRelY = std::clamp((TeamPos.y - Pos.y) / AI::MAX_DIST_SCALE, -1.0f, 1.0f);
		out_features.m_TeammateFrozen = 1.0f;
	}

	// 7. Previous Action Feedback (3)
	out_features.m_PrevDirection = (float)m_LastAction.m_Direction;
	out_features.m_PrevJump = m_LastAction.m_Jump ? 1.0f : 0.0f;
	out_features.m_PrevHook = m_LastAction.m_Hook ? 1.0f : 0.0f;
}

bool CAIEngineComponent::ProcessInput(CNetObj_PlayerInput *pInput, const CNetObj_PlayerInput &OrigInput, int DummyIndex)
{
	const bool Enabled = (g_Config.m_AiEnable != 0) || (g_Config.m_AiMode > 0);
	if(!Enabled || !m_pEngine)
		return false;

	// Synchronize configuration parameters
	auto &ctrl = m_pEngine->GetController();
	ctrl.SetSmoothAim(g_Config.m_AiSmoothAim != 0);
	ctrl.SetWallhopEnabled(g_Config.m_AiWallhop != 0);
	ctrl.SetSmartHookEnabled(g_Config.m_AiSmartHook != 0);
	ctrl.SetAutoUnfreezeEnabled(g_Config.m_AiAutoUnfreeze != 0);
	ctrl.SetFreezeAvoidanceEnabled(g_Config.m_AiFreezeAvoid != 0);
	ctrl.SetBlockMode(g_Config.m_AiBlockMode != 0);
	ctrl.SetSlingCombosEnabled(g_Config.m_AiSlingCombos != 0);
	ctrl.SetHammerFlickEnabled(g_Config.m_AiHammerFlick != 0);
	ctrl.SetHammerTimingEnabled(g_Config.m_AiHammerTiming != 0);
	ctrl.SetFreezeFinisherEnabled(g_Config.m_AiFreezeFinisher != 0);
	ctrl.SetFlyStrikeEnabled(g_Config.m_AiFlyStrike != 0);
	ctrl.SetLeadAimEnabled(g_Config.m_AiLeadAim != 0);
	ctrl.SetAimSensitivity((float)g_Config.m_AiAimSens);
	ctrl.SetKNNBlendWeight((float)g_Config.m_AiKnnBlend / 100.0f);
	ctrl.SetConfidenceThreshold((float)g_Config.m_AiConfidence / 100.0f);

	const AI::EAIMode Mode = (AI::EAIMode)(g_Config.m_AiEnable ? (g_Config.m_AiMode == 0 ? 1 : g_Config.m_AiMode) : g_Config.m_AiMode);
	ctrl.SetMode(Mode);
	m_pEngine->SetMode(Mode);

	if(Mode == AI::EAIMode::OFF)
		return false;

	AI::AIFeatures state;
	ExtractFeatures(DummyIndex, state);

	AI::AIAction act = m_pEngine->PredictTick(state, Client()->RenderFrameTime());
	m_LastAction = act;

	int LocalId = GameClient()->m_Snap.m_LocalClientId;

	// Live Online Observation: record current state & executed action into replay memory
	if(Client()->State() == IClient::STATE_ONLINE && LocalId >= 0)
	{
		AI::AIAction executed = act;
		if(Mode == AI::EAIMode::ASSIST || Mode == AI::EAIMode::LEARN_LIVE)
		{
			executed.m_Direction = OrigInput.m_Direction;
			executed.m_Jump = (OrigInput.m_Jump != 0);
			executed.m_Hook = (OrigInput.m_Hook != 0);
			executed.m_Fire = (OrigInput.m_Fire != 0);
			executed.m_TargetX = (float)OrigInput.m_TargetX;
			executed.m_TargetY = (float)OrigInput.m_TargetY;
		}

		m_pEngine->GetLiveObserver().ObserveTick(Client()->GameTick(DummyIndex), LocalId, state, executed);
	}

	if(Mode == AI::EAIMode::AUTOPILOT || g_Config.m_AiEnable)
	{
		pInput->m_Direction = act.m_Direction;
		pInput->m_Jump = act.m_Jump ? 1 : 0;
		pInput->m_Hook = act.m_Hook ? 1 : 0;

		// Proper Teeworlds network protocol fire press/release handling:
		// Odd = Key Pressed, Even = Key Released
		if(act.m_Fire)
		{
			if((pInput->m_Fire & 1) == 0)
				pInput->m_Fire++;
			pInput->m_Fire |= 1;
		}
		else
		{
			if((pInput->m_Fire & 1) != 0)
				pInput->m_Fire++;
			pInput->m_Fire &= ~1;
		}

		if(act.m_TargetX == 0.0f && act.m_TargetY == 0.0f)
		{
			act.m_TargetX = 100.0f;
		}
		pInput->m_TargetX = (int)act.m_TargetX;
		pInput->m_TargetY = (int)act.m_TargetY;

		// Move in-game visual crosshair to track target
		GameClient()->m_Controls.m_aMousePos[DummyIndex] = vec2(act.m_TargetX, act.m_TargetY);

		if(g_Config.m_AiBlockMode || act.m_WantedWeapon >= 0)
		{
			pInput->m_WantedWeapon = (act.m_WantedWeapon >= 0 ? act.m_WantedWeapon : 0) + 1; // 1 = Hammer
		}
		return true;
	}
	else if(Mode == AI::EAIMode::ASSIST)
	{
		// Assist mode: auto-unfreeze or close-combat hammer assist
		if(act.m_Fire || (act.m_Hook && state.m_TeammateFrozen > 0.5f) || (act.m_Tactics == AI::EBlockTactics::DEFENSIVE_RECOVERY))
		{
			if(act.m_Fire)
			{
				if((pInput->m_Fire & 1) == 0)
					pInput->m_Fire++;
				pInput->m_Fire |= 1;
			}
			else
			{
				if((pInput->m_Fire & 1) != 0)
					pInput->m_Fire++;
				pInput->m_Fire &= ~1;
			}
			pInput->m_TargetX = (int)act.m_TargetX;
			pInput->m_TargetY = (int)act.m_TargetY;
			GameClient()->m_Controls.m_aMousePos[DummyIndex] = vec2(act.m_TargetX, act.m_TargetY);
			if(act.m_WantedWeapon >= 0) pInput->m_WantedWeapon = act.m_WantedWeapon + 1;
			return true;
		}
	}

	return false;
}

void CAIEngineComponent::OnUpdate()
{
	if(Client()->State() != IClient::STATE_ONLINE || !m_pEngine)
		return;

	int LocalId = GameClient()->m_Snap.m_LocalClientId;
	if(LocalId >= 0 && LocalId < MAX_CLIENTS && GameClient()->m_Snap.m_aCharacters[LocalId].m_Active)
	{
		CCharacter *pChar = GameClient()->m_RegularPredictedWorld.GetCharacterById(LocalId);
		bool IsFrozen = (pChar && (pChar->Core()->m_IsInFreeze || pChar->m_FreezeTime > 0)) ||
		                GameClient()->m_aClients[LocalId].m_DeepFrozen ||
		                GameClient()->m_aClients[LocalId].m_LiveFrozen;

		// Anti-Freeze Self-Correction: trigger penalty gradient if transitioning into freeze
		if(!m_WasFrozen && IsFrozen)
		{
			m_pEngine->GetLiveObserver().ReinforceAntiFreeze(35);
			SetStatus("Анти-Фриз коррекция: нейросеть обучается избегать ловушки");
		}
		m_WasFrozen = IsFrozen;
	}

	// Real-Time Online Learning Step (continuous gradient descent)
	m_OnlineTrainCounter++;
	if(m_OnlineTrainCounter >= 4)
	{
		m_OnlineTrainCounter = 0;
		if(g_Config.m_AiLearnLive || g_Config.m_AiMode > 0 || g_Config.m_AiEnable)
		{
			m_pEngine->UpdateLiveLearning(0.0008f);
		}
	}
}

void CAIEngineComponent::OnNewSnapshot()
{
	// Also observe other active players on the server
	if(Client()->State() != IClient::STATE_ONLINE || !m_pEngine || !g_Config.m_AiLearnLive)
		return;

	int LocalId = GameClient()->m_Snap.m_LocalClientId;
	for(int i = 0; i < MAX_CLIENTS; ++i)
	{
		if(i == LocalId || !GameClient()->m_Snap.m_aCharacters[i].m_Active)
			continue;

		// Capture moving player transitions occasionally to diversify training
		if(rand() % 10 == 0)
		{
			AI::AIFeatures obs_state;
			const vec2 Pos = GameClient()->m_aClients[i].m_RegularPredicted.m_Pos;
			const vec2 Vel = GameClient()->m_aClients[i].m_RegularPredicted.m_Vel;

			obs_state.m_VelX = std::clamp(Vel.x / AI::MAX_VEL_SCALE, -1.0f, 1.0f);
			obs_state.m_VelY = std::clamp(Vel.y / AI::MAX_VEL_SCALE, -1.0f, 1.0f);
			obs_state.m_Speed = std::clamp(length(Vel) / AI::MAX_VEL_SCALE, 0.0f, 1.0f);

			AI::AIAction obs_act;
			obs_act.m_Direction = (Vel.x > 30.0f) ? 1 : ((Vel.x < -30.0f) ? -1 : 0);
			obs_act.m_Jump = (Vel.y < -300.0f);

			m_pEngine->GetLiveObserver().ObserveTick(Client()->GameTick(g_Config.m_ClDummy), i, obs_state, obs_act);
		}
	}
}

void CAIEngineComponent::OnMessage(int MsgType, void *pRawMsg)
{
}

void CAIEngineComponent::OnRender()
{
	if(!g_Config.m_AiHud || Client()->State() != IClient::STATE_ONLINE)
		return;

	Graphics()->MapScreenToSize(800.0f, 600.0f);

	const float BoxWidth = 270.0f;
	const float BoxHeight = 125.0f;
	const float BoxX = 800.0f - BoxWidth - 10.0f;
	const float BoxY = 10.0f;

	// Dark semi-transparent card background
	Graphics()->BlendNormal();
	Graphics()->TextureClear();
	Graphics()->QuadsBegin();
	Graphics()->SetColor(0.03f, 0.06f, 0.10f, 0.90f);
	IGraphics::CQuadItem Quad(BoxX, BoxY, BoxWidth, BoxHeight);
	Graphics()->QuadsDrawTL(&Quad, 1);

	// Accent cyan top border
	Graphics()->SetColor(0.0f, 0.85f, 1.0f, 0.95f);
	IGraphics::CQuadItem TopBorder(BoxX, BoxY, BoxWidth, 2.5f);
	Graphics()->QuadsDrawTL(&TopBorder, 1);
	Graphics()->QuadsEnd();

	// Telemetry Texts in Russian
	char aBuf[128];
	const char *pModeStr = (g_Config.m_AiEnable || g_Config.m_AiMode == 1) ? "АВТОПИЛОТ (БЛОКЕР)" : (g_Config.m_AiMode == 2 ? "АССИСТЕНТ" : (g_Config.m_AiMode == 3 ? "ОБУЧЕНИЕ" : "ВЫКЛ"));
	str_format(aBuf, sizeof(aBuf), "NeuroBlock AI: %s", pModeStr);
	TextRender()->Text(BoxX + 8.0f, BoxY + 6.0f, 11.5f, aBuf, -1.0f);

	const char *pTacticStr = "СПЕЙСИНГ И НАВЕДЕНИЕ";
	if(m_LastAction.m_Tactics == AI::EBlockTactics::FREEZE_SLAM) pTacticStr = "ДОБИВАНИЕ ВО ФРИЗЕ";
	else if(m_LastAction.m_Tactics == AI::EBlockTactics::FLY_STRIKE) pTacticStr = "ЗАЛЁТНЫЙ УДАР";
	else if(m_LastAction.m_Tactics == AI::EBlockTactics::OFFENSIVE_DRAG) pTacticStr = "СЛИНГ-БРОСОК В ФРИЗ";
	else if(m_LastAction.m_Tactics == AI::EBlockTactics::HAMMER_EXECUTE) pTacticStr = "ФЛИК МОЛОТОМ";
	else if(m_LastAction.m_Tactics == AI::EBlockTactics::DEFENSIVE_RECOVERY) pTacticStr = "АНТИ-ФРИЗ СПАСЕНИЕ";
	else if(m_LastAction.m_Tactics == AI::EBlockTactics::TEAM_RESCUE) pTacticStr = "РАЗМОРОЗКА СОЮЗНИКА";

	str_format(aBuf, sizeof(aBuf), "Тактика: [%s]", pTacticStr);
	TextRender()->Text(BoxX + 8.0f, BoxY + 23.0f, 9.5f, aBuf, -1.0f);

	const auto &stats = m_pEngine->GetStats();
	str_format(aBuf, sizeof(aBuf), "Точность: %.1f%% | Движ=%+d Прыж=%d Хук=%d",
		stats.m_DirectionAccuracy * 100.0f,
		m_LastAction.m_Direction,
		m_LastAction.m_Jump ? 1 : 0,
		m_LastAction.m_Hook ? 1 : 0);
	TextRender()->Text(BoxX + 8.0f, BoxY + 40.0f, 9.0f, aBuf, -1.0f);

	str_format(aBuf, sizeof(aBuf), "Модули: [Тайминг:%s Залёт:%s Фриз:%s]",
		g_Config.m_AiHammerTiming ? "ВКЛ" : "ВЫКЛ",
		g_Config.m_AiFlyStrike ? "ВКЛ" : "ВЫКЛ",
		g_Config.m_AiFreezeFinisher ? "ВКЛ" : "ВЫКЛ");
	TextRender()->Text(BoxX + 8.0f, BoxY + 57.0f, 8.5f, aBuf, -1.0f);

	if(m_aStatusMsg[0] && (time_get() - m_StatusMsgTime < time_freq() * 5))
	{
		TextRender()->Text(BoxX + 8.0f, BoxY + 77.0f, 8.5f, m_aStatusMsg, -1.0f);
	}
	else
	{
		str_format(aBuf, sizeof(aBuf), "Семплов: %llu | Обучение: Активно", (unsigned long long)stats.m_TotalSamplesRecorded);
		TextRender()->Text(BoxX + 8.0f, BoxY + 77.0f, 8.5f, aBuf, -1.0f);
	}
}
