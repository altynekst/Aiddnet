#include "menus.h"
#include <base/str.h>
#include <game/client/gameclient.h>
#include <game/client/ui.h>
#include <game/client/components/ai_engine_component.h>
#include <game/localization.h>
#include <engine/shared/config.h>
#include <engine/textrender.h>
#include <engine/graphics.h>

void CMenus::RenderSettingsAI(CUIRect MainView)
{
	CUIRect Left, Right, Label, Button, Box, Row;
	char aBuf[256];

	MainView.VSplitMid(&Left, &Right, 16.0f);

	// ==========================================
	// LEFT COLUMN: Modes & Combat Reflexes
	// ==========================================
	{
		// 1. Master Switch & Mode Box
		Left.HSplitTop(140.0f, &Box, &Left);
		Box.Draw(ColorRGBA(0.0f, 0.0f, 0.0f, 0.25f), IGraphics::CORNER_ALL, 5.0f);
		Box.Margin(8.0f, &Box);

		Box.HSplitTop(20.0f, &Label, &Box);
		Ui()->DoLabel(&Label, "Режимы работы NeuroBlock", 15.0f, TEXTALIGN_ML);
		Box.HSplitTop(4.0f, nullptr, &Box);

		// Enable AI Autopilot Checkbox
		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiEnable, "Включить автопилот (AI управляет персонажем)", g_Config.m_AiEnable, &Button))
		{
			g_Config.m_AiEnable ^= 1;
			if(g_Config.m_AiEnable)
				g_Config.m_AiMode = 1;
		}

		// 3-Segment Mode Buttons
		Box.HSplitTop(4.0f, nullptr, &Box);
		Box.HSplitTop(22.0f, &Button, &Box);
		CUIRect ModeBtn;
		const float BtnW = Button.w / 3.0f;

		static CButtonContainer s_ModeAutopilot, s_ModeAssist, s_ModeLearn;
		Button.VSplitLeft(BtnW, &ModeBtn, &Button);
		if(DoButton_Menu(&s_ModeAutopilot, "Автопилот", g_Config.m_AiMode == 1, &ModeBtn, BUTTONFLAG_LEFT, nullptr, IGraphics::CORNER_L))
		{
			g_Config.m_AiMode = 1;
			g_Config.m_AiEnable = 1;
		}

		Button.VSplitLeft(BtnW, &ModeBtn, &Button);
		if(DoButton_Menu(&s_ModeAssist, "Ассистент", g_Config.m_AiMode == 2, &ModeBtn, BUTTONFLAG_LEFT, nullptr, IGraphics::CORNER_NONE))
		{
			g_Config.m_AiMode = 2;
			g_Config.m_AiEnable = 0;
		}

		if(DoButton_Menu(&s_ModeLearn, "Дообучение", g_Config.m_AiMode == 3, &Button, BUTTONFLAG_LEFT, nullptr, IGraphics::CORNER_R))
		{
			g_Config.m_AiMode = 3;
			g_Config.m_AiEnable = 0;
			g_Config.m_AiLearnLive = 1;
		}

		Box.HSplitTop(5.0f, nullptr, &Box);
		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiHud, "Показывать боевой HUD и телеметрию", g_Config.m_AiHud, &Button))
		{
			g_Config.m_AiHud ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiLearnLive, "Непрерывное онлайн-дообучение во время матча", g_Config.m_AiLearnLive, &Button))
		{
			g_Config.m_AiLearnLive ^= 1;
		}

		Left.HSplitTop(8.0f, nullptr, &Left);

		// 2. High-Skill Reflex & Block Tactics Box
		Left.HSplitTop(295.0f, &Box, &Left);
		Box.Draw(ColorRGBA(0.0f, 0.0f, 0.0f, 0.25f), IGraphics::CORNER_ALL, 5.0f);
		Box.Margin(8.0f, &Box);

		Box.HSplitTop(20.0f, &Label, &Box);
		Ui()->DoLabel(&Label, "Боевые тактики и рефлексы Блокера", 15.0f, TEXTALIGN_ML);
		Box.HSplitTop(4.0f, nullptr, &Box);

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiBlockMode, "Боевой движок NeuroBlock (Режим Блокера)", g_Config.m_AiBlockMode, &Button))
		{
			g_Config.m_AiBlockMode ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiHammerTiming, "Умный тайминг молота (Удары в тайминг без спама)", g_Config.m_AiHammerTiming, &Button))
		{
			g_Config.m_AiHammerTiming ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiFreezeFinisher, "Добивание во фризе (Затаскивать и выбивать в яму)", g_Config.m_AiFreezeFinisher, &Button))
		{
			g_Config.m_AiFreezeFinisher ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiFlyStrike, "Залётные удары с разгона (Fly-in Hammer Strike)", g_Config.m_AiFlyStrike, &Button))
		{
			g_Config.m_AiFlyStrike ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiSlingCombos, "Комбо-броски через себя (Sling Drag в яму)", g_Config.m_AiSlingCombos, &Button))
		{
			g_Config.m_AiSlingCombos ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiLeadAim, "Баллистическое упреждение хука (Lead Aim)", g_Config.m_AiLeadAim, &Button))
		{
			g_Config.m_AiLeadAim ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiSmartHook, "Умный 16-лучевой зацеп за стены (Smart Hook)", g_Config.m_AiSmartHook, &Button))
		{
			g_Config.m_AiSmartHook ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiWallhop, "Скалолазание и прыжки от стен (Wallhop)", g_Config.m_AiWallhop, &Button))
		{
			g_Config.m_AiWallhop ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiFreezeAvoid, "Анти-Фриз: радар опасных зон и ям с лавой", g_Config.m_AiFreezeAvoid, &Button))
		{
			g_Config.m_AiFreezeAvoid ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiAutoUnfreeze, "Автоматическая разморозка союзников молотом", g_Config.m_AiAutoUnfreeze, &Button))
		{
			g_Config.m_AiAutoUnfreeze ^= 1;
		}

		Box.HSplitTop(20.0f, &Button, &Box);
		if(DoButton_CheckBox(&g_Config.m_AiSmoothAim, "Плавное человекоподобное сглаживание аима", g_Config.m_AiSmoothAim, &Button))
		{
			g_Config.m_AiSmoothAim ^= 1;
		}
	}

	// ==========================================
	// RIGHT COLUMN: Tuning, Brain Management & Stats
	// ==========================================
	{
		// 1. Fine-Tuning Sliders Box
		Right.HSplitTop(130.0f, &Box, &Right);
		Box.Draw(ColorRGBA(0.0f, 0.0f, 0.0f, 0.25f), IGraphics::CORNER_ALL, 5.0f);
		Box.Margin(8.0f, &Box);

		Box.HSplitTop(20.0f, &Label, &Box);
		Ui()->DoLabel(&Label, "Тонкая настройка параметров", 15.0f, TEXTALIGN_ML);
		Box.HSplitTop(4.0f, nullptr, &Box);

		Box.HSplitTop(20.0f, &Button, &Box);
		Ui()->DoScrollbarOption(&g_Config.m_AiAimSens, &g_Config.m_AiAimSens, &Button,
			"Скорость наводки прицела", 1, 50, &CUi::ms_LinearScrollbarScale, 0);

		Box.HSplitTop(20.0f, &Button, &Box);
		Ui()->DoScrollbarOption(&g_Config.m_AiKnnBlend, &g_Config.m_AiKnnBlend, &Button,
			"Смешивание с памятью KD-Tree %", 0, 100, &CUi::ms_LinearScrollbarScale, 0);

		Box.HSplitTop(20.0f, &Button, &Box);
		Ui()->DoScrollbarOption(&g_Config.m_AiConfidence, &g_Config.m_AiConfidence, &Button,
			"Порог уверенности решений %", 1, 100, &CUi::ms_LinearScrollbarScale, 0);

		Right.HSplitTop(8.0f, nullptr, &Right);

		// 2. Training & Brain Operations Box
		Right.HSplitTop(115.0f, &Box, &Right);
		Box.Draw(ColorRGBA(0.0f, 0.0f, 0.0f, 0.25f), IGraphics::CORNER_ALL, 5.0f);
		Box.Margin(8.0f, &Box);

		Box.HSplitTop(20.0f, &Label, &Box);
		Ui()->DoLabel(&Label, "База знаний и веса нейросети", 15.0f, TEXTALIGN_ML);
		Box.HSplitTop(6.0f, nullptr, &Box);

		Box.HSplitTop(24.0f, &Row, &Box);
		static CButtonContainer s_BtnTrain, s_BtnSave;
		Row.VSplitMid(&Button, &Row, 8.0f);

		if(DoButton_Menu(&s_BtnTrain, "Обучить на всех демках", 0, &Button, BUTTONFLAG_LEFT, nullptr, IGraphics::CORNER_ALL))
		{
			GameClient()->m_AIEngineComponent.TrainOnDemos();
		}

		if(DoButton_Menu(&s_BtnSave, "Сохранить модель (.bin)", 0, &Row, BUTTONFLAG_LEFT, nullptr, IGraphics::CORNER_ALL))
		{
			GameClient()->m_AIEngineComponent.SaveModel("ddnet_ai_model.bin");
		}

		Box.HSplitTop(6.0f, nullptr, &Box);
		Box.HSplitTop(24.0f, &Row, &Box);
		static CButtonContainer s_BtnLoad, s_BtnReset;
		Row.VSplitMid(&Button, &Row, 8.0f);

		if(DoButton_Menu(&s_BtnLoad, "Загрузить модель (.bin)", 0, &Button, BUTTONFLAG_LEFT, nullptr, IGraphics::CORNER_ALL))
		{
			GameClient()->m_AIEngineComponent.LoadModel("ddnet_ai_model.bin");
		}

		if(DoButton_Menu(&s_BtnReset, "Сбросить веса нейросети", 0, &Row, BUTTONFLAG_LEFT, nullptr, IGraphics::CORNER_ALL))
		{
			GameClient()->m_AIEngineComponent.ResetModel();
		}

		Right.HSplitTop(8.0f, nullptr, &Right);

		// 3. Diagnostics & Telemetry Card
		Right.HSplitTop(140.0f, &Box, &Right);
		Box.Draw(ColorRGBA(0.02f, 0.06f, 0.12f, 0.75f), IGraphics::CORNER_ALL, 5.0f);
		Box.Margin(8.0f, &Box);

		Box.HSplitTop(20.0f, &Label, &Box);
		Ui()->DoLabel(&Label, "Диагностика и статус нейросети", 14.0f, TEXTALIGN_ML);
		Box.HSplitTop(4.0f, nullptr, &Box);

		const auto &stats = GameClient()->m_AIEngineComponent.GetEngine()->GetStats();

		Box.HSplitTop(18.0f, &Label, &Box);
		str_format(aBuf, sizeof(aBuf), "Обработано демок: %d   |   Семплов в памяти: %llu",
			stats.m_TotalDemosParsed, (unsigned long long)stats.m_TotalSamplesRecorded);
		Ui()->DoLabel(&Label, aBuf, 11.5f, TEXTALIGN_ML);

		Box.HSplitTop(18.0f, &Label, &Box);
		str_format(aBuf, sizeof(aBuf), "Точность: Движение: %.1f%%  Прыжки: %.1f%%  Хук: %.1f%%",
			stats.m_DirectionAccuracy * 100.0f, stats.m_JumpAccuracy * 100.0f, stats.m_HookAccuracy * 100.0f);
		Ui()->DoLabel(&Label, aBuf, 11.5f, TEXTALIGN_ML);

		Box.HSplitTop(18.0f, &Label, &Box);
		str_format(aBuf, sizeof(aBuf), "Ошибка обучения (Loss): %.4f   |   Инференс: %.1f мкс",
			stats.m_RecentLoss, stats.m_LastInferenceMicroseconds);
		Ui()->DoLabel(&Label, aBuf, 11.0f, TEXTALIGN_ML);

		Box.HSplitTop(20.0f, &Label, &Box);
		const char *pStatus = GameClient()->m_AIEngineComponent.GetStatusMessage();
		if(!pStatus || !pStatus[0])
			pStatus = GameClient()->m_AIEngineComponent.IsModelLoaded() ? "Статус: Нейросеть активна (Модель обучена)" : "Статус: Нейросеть готова к работе";
		Ui()->DoLabel(&Label, pStatus, 11.5f, TEXTALIGN_ML);
	}
}
