#ifndef GAME_CLIENT_COMPONENTS_AI_ENGINE_COMPONENT_H
#define GAME_CLIENT_COMPONENTS_AI_ENGINE_COMPONENT_H

#include <engine/console.h>
#include <game/client/component.h>
#include <ai/ai_engine.h>
#include <memory>
#include <string>

struct CNetObj_PlayerInput;

class CAIEngineComponent : public CComponent
{
public:
	CAIEngineComponent();
	~CAIEngineComponent() override;

	int Sizeof() const override { return sizeof(*this); }
	void OnInit() override;
	void OnReset() override;
	void OnConsoleInit() override;
	void OnRender() override;
	void OnUpdate() override;
	void OnNewSnapshot() override;
	void OnMessage(int MsgType, void *pRawMsg) override;

	// Overwrite outgoing player input when AI is active
	bool ProcessInput(CNetObj_PlayerInput *pInput, const CNetObj_PlayerInput &OrigInput, int DummyIndex);

	AI::AIEngine *GetEngine() { return m_pEngine.get(); }
	const AI::AIEngine *GetEngine() const { return m_pEngine.get(); }

	void ExtractFeatures(int DummyIndex, AI::AIFeatures &out_features) const;

	// High-level operations triggered by UI or Console
	void TrainOnDemos();
	void SaveModel(const char *pFilename = nullptr);
	void LoadModel(const char *pFilename = nullptr);
	void ResetModel();

	const char *GetStatusMessage() const { return m_aStatusMsg; }
	bool IsModelLoaded() const { return m_ModelLoaded; }

private:
	std::unique_ptr<AI::AIEngine> m_pEngine;
	AI::AIAction m_LastAction;
	bool m_ModelLoaded = false;
	bool m_WasFrozen = false;
	int m_OnlineTrainCounter = 0;
	int m_FireCounter = 0;
	mutable int m_LockedEnemyId = -1;
	char m_aStatusMsg[256] = {0};
	int64_t m_StatusMsgTime = 0;

	void SetStatus(const char *pMsg);
	std::string GetDefaultModelPath() const;

	static void ConTrainDemos(IConsole::IResult *pResult, void *pUserData);
	static void ConSaveModel(IConsole::IResult *pResult, void *pUserData);
	static void ConLoadModel(IConsole::IResult *pResult, void *pUserData);
	static void ConResetModel(IConsole::IResult *pResult, void *pUserData);
	static void ConStatus(IConsole::IResult *pResult, void *pUserData);
};

#endif // GAME_CLIENT_COMPONENTS_AI_ENGINE_COMPONENT_H
