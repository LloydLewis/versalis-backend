#include "VirtualHumanConversationComponent.h"

#include "Engine/GameInstance.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#include "ACEASRSubsystem.h"
#include "ACEASRBlueprintLibrary.h"
#include "ACEASRTypes.h"
#include "ACETTSSubsystem.h"

#include "Audio2FaceParameters.h"
#include "ACETypes.h"
#include "AsyncActionAnimateCharacter.h"

DEFINE_LOG_CATEGORY_STATIC(LogVirtualHuman, Log, All);

namespace
{
	TWeakObjectPtr<UVirtualHumanConversationComponent> GActiveVirtualHuman;

	/**
	 * The only channel back from the LLM is a plain reply string (SendACEASRLLMMessage's
	 * OnSuccess delegate is just choices[0].message.content - no room for a structured
	 * "should I end this?" field), so aisegment.py/server.py signal "end the conversation"
	 * by appending this marker to the reply text. Stripped here before the line is ever
	 * spoken or shown, so it's never audible/visible - it's purely a control signal.
	 */
	const FString EndConversationMarker = TEXT("[[END_SESSION]]");

	/** Detects and removes EndConversationMarker from a reply, leaving the spoken/displayed text clean. */
	FString StripEndConversationMarker(const FString& Content, bool& bOutFoundMarker)
	{
		bOutFoundMarker = Content.Contains(EndConversationMarker);
		if (!bOutFoundMarker)
		{
			return Content;
		}
		FString Cleaned = Content.Replace(*EndConversationMarker, TEXT(""));
		Cleaned.TrimStartAndEndInline();
		return Cleaned;
	}

	/** Wraps raw 16-bit PCM in a minimal RIFF/WAVE header and writes it to disk. */
	bool WritePcm16ToWavFile(const TArray<int16>& Pcm, int32 SampleRate, int32 NumChannels, const FString& FilePath)
	{
		if (Pcm.Num() == 0 || SampleRate <= 0 || NumChannels <= 0)
		{
			return false;
		}

		constexpr int32 BitsPerSample = 16;
		const int32 DataSize = Pcm.Num() * sizeof(int16);
		const int32 ByteRate = SampleRate * NumChannels * (BitsPerSample / 8);
		const int16 BlockAlign = static_cast<int16>(NumChannels * (BitsPerSample / 8));
		const int32 RiffChunkSize = 36 + DataSize;
		const int32 FmtChunkSize = 16;
		const int16 AudioFormatPCM = 1;
		const int16 NumChannelsI16 = static_cast<int16>(NumChannels);
		const int16 BitsPerSampleI16 = static_cast<int16>(BitsPerSample);

		TArray<uint8> Bytes;
		Bytes.Reserve(44 + DataSize);

		auto AppendTag = [&Bytes](const ANSICHAR* Tag)
		{
			Bytes.Append(reinterpret_cast<const uint8*>(Tag), 4);
		};
		auto AppendValue = [&Bytes](const auto& Value)
		{
			Bytes.Append(reinterpret_cast<const uint8*>(&Value), sizeof(Value));
		};

		AppendTag("RIFF");
		AppendValue(RiffChunkSize);
		AppendTag("WAVE");
		AppendTag("fmt ");
		AppendValue(FmtChunkSize);
		AppendValue(AudioFormatPCM);
		AppendValue(NumChannelsI16);
		AppendValue(SampleRate);
		AppendValue(ByteRate);
		AppendValue(BlockAlign);
		AppendValue(BitsPerSampleI16);
		AppendTag("data");
		AppendValue(DataSize);
		Bytes.Append(reinterpret_cast<const uint8*>(Pcm.GetData()), DataSize);

		IFileManager::Get().MakeDirectory(*FPaths::GetPath(FilePath), true);
		return FFileHelper::SaveArrayToFile(Bytes, *FilePath);
	}
}

UVirtualHumanConversationComponent::UVirtualHumanConversationComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UVirtualHumanConversationComponent::BeginPlay()
{
	Super::BeginPlay();

	GActiveVirtualHuman = this;

	UGameInstance* GameInstance = GetWorld() ? GetWorld()->GetGameInstance() : nullptr;
	if (!GameInstance)
	{
		UE_LOG(LogVirtualHuman, Error, TEXT("VirtualHumanConversationComponent: no GameInstance, cannot resolve ACE subsystems"));
		return;
	}

	ASRSubsystem = GameInstance->GetSubsystem<UACEASRSubsystem>();
	TTSSubsystem = GameInstance->GetSubsystem<UACETTSSubsystem>();

	if (ASRSubsystem)
	{
		ASRSubsystem->OnASRReady.AddDynamic(this, &UVirtualHumanConversationComponent::HandleASRReady);
		ASRSubsystem->OnASRError.AddDynamic(this, &UVirtualHumanConversationComponent::HandleASRError);
		ASRSubsystem->OnTranscriptFinalized.AddDynamic(this, &UVirtualHumanConversationComponent::HandleTranscriptFinalized);

		if (!ASRSubsystem->IsASRInitialized() && !ASRSubsystem->IsASRInitializing())
		{
			ASRSubsystem->InitializeASRAsync();
		}
	}
	else
	{
		UE_LOG(LogVirtualHuman, Error, TEXT("VirtualHumanConversationComponent: ACE ASR subsystem not found - is the ACE_ASR plugin enabled?"));
	}

	if (TTSSubsystem)
	{
		TTSSubsystem->OnTTSReady.AddDynamic(this, &UVirtualHumanConversationComponent::HandleTTSReady);
		TTSSubsystem->OnTTSError.AddDynamic(this, &UVirtualHumanConversationComponent::HandleTTSError);
		TTSSubsystem->OnSpeechFinished.AddDynamic(this, &UVirtualHumanConversationComponent::HandleSpeechFinished);
		TTSSubsystem->OnAudioSamplesReady.AddDynamic(this, &UVirtualHumanConversationComponent::HandleAudioSamplesReady);

		if (!TTSSubsystem->IsTTSInitialized() && !TTSSubsystem->IsTTSInitializing())
		{
			TTSSubsystem->InitializeTTSAsync();
		}
	}
	else
	{
		UE_LOG(LogVirtualHuman, Error, TEXT("VirtualHumanConversationComponent: ACE TTS subsystem not found - is the ACE_TTS plugin enabled?"));
	}
}

void UVirtualHumanConversationComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (ASRSubsystem)
	{
		ASRSubsystem->OnASRReady.RemoveDynamic(this, &UVirtualHumanConversationComponent::HandleASRReady);
		ASRSubsystem->OnASRError.RemoveDynamic(this, &UVirtualHumanConversationComponent::HandleASRError);
		ASRSubsystem->OnTranscriptFinalized.RemoveDynamic(this, &UVirtualHumanConversationComponent::HandleTranscriptFinalized);
	}

	if (TTSSubsystem)
	{
		TTSSubsystem->OnTTSReady.RemoveDynamic(this, &UVirtualHumanConversationComponent::HandleTTSReady);
		TTSSubsystem->OnTTSError.RemoveDynamic(this, &UVirtualHumanConversationComponent::HandleTTSError);
		TTSSubsystem->OnSpeechFinished.RemoveDynamic(this, &UVirtualHumanConversationComponent::HandleSpeechFinished);
		TTSSubsystem->OnAudioSamplesReady.RemoveDynamic(this, &UVirtualHumanConversationComponent::HandleAudioSamplesReady);
	}

	if (GActiveVirtualHuman.Get() == this)
	{
		GActiveVirtualHuman.Reset();
	}

	Super::EndPlay(EndPlayReason);
}

UVirtualHumanConversationComponent* UVirtualHumanConversationComponent::GetActiveVirtualHuman(const UObject* WorldContextObject)
{
	(void)WorldContextObject;
	UVirtualHumanConversationComponent* Result = GActiveVirtualHuman.Get();
	// Temporary diagnostic - narrowing down a repro where a second push-to-talk after
	// one full turn produces zero LogVirtualHuman output at all.
	UE_LOG(LogVirtualHuman, Log, TEXT("GetActiveVirtualHuman() -> %s"),
		Result ? *Result->GetOwner()->GetName() : TEXT("nullptr (GActiveVirtualHuman is stale/unset)"));
	return Result;
}

void UVirtualHumanConversationComponent::StartListening()
{
	// Temporary diagnostic - see GetActiveVirtualHuman() above.
	UE_LOG(LogVirtualHuman, Log, TEXT("StartListening() entered (state=%s, ASRSubsystem=%s, ASRInitialized=%s)"),
		*StaticEnum<EVirtualHumanState>()->GetNameStringByValue((int64)CurrentState),
		ASRSubsystem ? TEXT("valid") : TEXT("NULL"),
		(ASRSubsystem && ASRSubsystem->IsASRInitialized()) ? TEXT("true") : TEXT("false"));

	if (!ASRSubsystem)
	{
		OnConversationError.Broadcast(TEXT("ACE ASR subsystem unavailable"));
		return;
	}

	if (!ASRSubsystem->IsASRInitialized())
	{
		OnConversationError.Broadcast(TEXT("ACE ASR is not ready yet"));
		return;
	}

	if (CurrentState != EVirtualHumanState::Idle)
	{
		// Already listening, or busy thinking/speaking - ignore.
		UE_LOG(LogVirtualHuman, Log, TEXT("StartListening ignored - current state is %s, not Idle"),
			*StaticEnum<EVirtualHumanState>()->GetNameStringByValue((int64)CurrentState));
		return;
	}

	int32 DeviceIndexToUse = PreferredCaptureDeviceIndex;
	if (DeviceIndexToUse < 0)
	{
		// Don't rely on WASAPI's "default device" role resolution - it's easy for that
		// to come back empty (e.g. Bluetooth/USB headsets after a reconnect) even though
		// the device itself enumerates fine. Ask the plugin directly for what it can see
		// and pick the first entry instead.
		const TArray<FACEASRAudioInputDeviceInfo> Devices = ASRSubsystem->GetAvailableInputDevices();
		if (Devices.Num() > 0)
		{
			DeviceIndexToUse = Devices[0].DeviceIndex;
		}
	}

	if (ASRSubsystem->StartMicrophoneTranscription(DeviceIndexToUse))
	{
		if (!bConversationInProgress)
		{
			bConversationInProgress = true;
			ConsecutiveTurnCount = 0;
			OnConversationStarted.Broadcast();
		}
		SetState(EVirtualHumanState::Listening);
	}
	else
	{
		OnConversationError.Broadcast(TEXT("Failed to start microphone capture"));
	}
}

void UVirtualHumanConversationComponent::StopListening()
{
	if (!ASRSubsystem || CurrentState != EVirtualHumanState::Listening)
	{
		return;
	}

	// OnTranscriptFinalized fires asynchronously once NVIGI latches the final transcript.
	ASRSubsystem->StopMicrophoneTranscription();
}

void UVirtualHumanConversationComponent::HandleASRReady()
{
	UE_LOG(LogVirtualHuman, Log, TEXT("ACE ASR ready"));

	if (ASRSubsystem)
	{
		const TArray<FACEASRAudioInputDeviceInfo> Devices = ASRSubsystem->GetAvailableInputDevices();
		UE_LOG(LogVirtualHuman, Log, TEXT("ACE ASR sees %d capture device(s):"), Devices.Num());
		for (const FACEASRAudioInputDeviceInfo& Device : Devices)
		{
			UE_LOG(LogVirtualHuman, Log, TEXT("  [%d] \"%s\" (Id=%s, Channels=%d, PreferredSampleRate=%d, HW AEC=%s)"),
				Device.DeviceIndex, *Device.DeviceName, *Device.DeviceId, Device.InputChannels,
				Device.PreferredSampleRate, Device.bSupportsHardwareAEC ? TEXT("true") : TEXT("false"));
		}
	}
}

void UVirtualHumanConversationComponent::HandleASRError(const FString& ErrorMessage)
{
	UE_LOG(LogVirtualHuman, Warning, TEXT("ACE ASR error: %s"), *ErrorMessage);
	OnConversationError.Broadcast(ErrorMessage);
	EndConversation();
}

void UVirtualHumanConversationComponent::HandleTranscriptFinalized(const FString& FinalTranscript)
{
	if (FinalTranscript.IsEmpty())
	{
		// NVIGI's inactivity watchdog finalized this turn with no speech in it (the user
		// paused without saying anything). That's not an LLM decision, so it doesn't end
		// the conversation - if one is already running, just keep listening for them.
		SetState(EVirtualHumanState::Idle);
		if (bConversationInProgress)
		{
			StartListening();
		}
		return;
	}

	OnUserTranscript.Broadcast(FinalTranscript);
	SetState(EVirtualHumanState::Thinking);

	FOnACEASRLLMResponse SuccessDelegate;
	SuccessDelegate.BindDynamic(this, &UVirtualHumanConversationComponent::HandleLLMResponse);

	FOnACEASRLLMError ErrorDelegate;
	ErrorDelegate.BindDynamic(this, &UVirtualHumanConversationComponent::HandleLLMError);

	// SystemPrompt is intentionally left empty - the FastAPI server behind LLMBaseUrl
	// owns the system prompt (and any safety-critical instructions), so the game
	// client cannot override it.
	UACEASRBlueprintLibrary::SendACEASRLLMMessage(
		EACEASRLLMRole::User,
		TEXT(""),
		FinalTranscript,
		LLMBaseUrl,
		LLMModelId,
		LLMApiKey,
		FACEASRLLMPayloadOptions(),
		SuccessDelegate,
		ErrorDelegate);
}

void UVirtualHumanConversationComponent::HandleLLMResponse(const FString& Content)
{
	// aisegment.py/server.py decide whether this conversation is over and, if so, signal it
	// by embedding EndConversationMarker in the reply - strip it here so it's never spoken
	// or shown, and remember the decision until this line finishes playing.
	const FString CleanContent = StripEndConversationMarker(Content, bPendingEndConversation);

	OnAssistantReply.Broadcast(CleanContent);

	if (!TTSSubsystem)
	{
		OnConversationError.Broadcast(TEXT("ACE TTS subsystem unavailable"));
		EndConversation();
		return;
	}

	AccumulatedPCM.Reset();
	AccumulatedSampleRate = 0;
	AccumulatedChannels = 0;

	if (TTSSubsystem->SpeakTextAsync(CleanContent))
	{
		SetState(EVirtualHumanState::Speaking);
	}
	else
	{
		OnConversationError.Broadcast(TEXT("ACE TTS failed to start speaking"));
		EndConversation();
	}
}

void UVirtualHumanConversationComponent::HandleLLMError(const FString& ErrorMessage)
{
	UE_LOG(LogVirtualHuman, Warning, TEXT("LLM request failed: %s"), *ErrorMessage);
	OnConversationError.Broadcast(ErrorMessage);
	EndConversation();
}

void UVirtualHumanConversationComponent::HandleTTSReady()
{
	UE_LOG(LogVirtualHuman, Log, TEXT("ACE TTS ready"));
}

void UVirtualHumanConversationComponent::HandleTTSError(const FString& ErrorMessage)
{
	UE_LOG(LogVirtualHuman, Warning, TEXT("ACE TTS error: %s"), *ErrorMessage);
	OnConversationError.Broadcast(ErrorMessage);
	EndConversation();
}

void UVirtualHumanConversationComponent::HandleSpeechFinished(const FString& Text, bool bSucceeded)
{
	// Not authoritative for the Idle transition - see HandleAnimationSendCompleted.
	// TTS's own AudioComponent is volume-muted, and UE's audio engine can virtualize/skip
	// completion callbacks for sounds it judges inaudible, so this may fire late or not at all.
	if (!bSucceeded)
	{
		OnConversationError.Broadcast(TEXT("TTS speech was interrupted before finishing"));
	}
}

void UVirtualHumanConversationComponent::HandleAnimationSendCompleted(bool bSuccess)
{
	if (!bSuccess)
	{
		OnConversationError.Broadcast(TEXT("Audio2Face failed to process the synthesized speech"));
		EndConversation();
		return;
	}

	SetState(EVirtualHumanState::Idle);

	if (bPendingEndConversation)
	{
		UE_LOG(LogVirtualHuman, Log, TEXT("LLM signaled end of conversation"));
		EndConversation();
		return;
	}

	++ConsecutiveTurnCount;
	if (MaxConsecutiveTurns > 0 && ConsecutiveTurnCount >= MaxConsecutiveTurns)
	{
		UE_LOG(LogVirtualHuman, Warning,
			TEXT("Reached MaxConsecutiveTurns (%d) without the LLM ending the conversation - ending it as a safety backstop"),
			MaxConsecutiveTurns);
		EndConversation();
		return;
	}

	// The LLM didn't end things, so this is a real back-and-forth conversation - keep
	// listening for the user's next turn without waiting on more input from them.
	StartListening();
}

void UVirtualHumanConversationComponent::HandleAudioSamplesReady(TArray<int16> Samples, int32 SampleRate, int32 Channels, bool bIsFinalChunk)
{
	AccumulatedPCM.Append(Samples);
	AccumulatedSampleRate = SampleRate;
	AccumulatedChannels = Channels;

	if (bIsFinalChunk)
	{
		FlushAccumulatedAudioToLipsync();
	}
}

AActor* UVirtualHumanConversationComponent::ResolveLipsyncCharacter() const
{
	TArray<AActor*, TInlineAllocator<8>> Candidates;

	if (AActor* Primary = LipsyncCharacterOverride ? ToRawPtr(LipsyncCharacterOverride) : GetOwner())
	{
		Candidates.Add(Primary);
	}

	for (const TObjectPtr<AActor>& Extra : AdditionalVirtualHumans)
	{
		if (Extra)
		{
			Candidates.Add(Extra);
		}
	}

	if (Candidates.Num() == 0)
	{
		return nullptr;
	}

	AActor* Chosen = Candidates[FMath::RandHelper(Candidates.Num())];
	if (Candidates.Num() > 1)
	{
		UE_LOG(LogVirtualHuman, Log, TEXT("Randomly picked '%s' to deliver this reply (%d candidate(s))"),
			*GetNameSafe(Chosen), Candidates.Num());
	}
	return Chosen;
}

void UVirtualHumanConversationComponent::FlushAccumulatedAudioToLipsync()
{
	if (AccumulatedPCM.Num() == 0)
	{
		return;
	}

	AActor* Character = ResolveLipsyncCharacter();
	if (!Character)
	{
		OnConversationError.Broadcast(TEXT("No character configured for Audio2Face lipsync"));
		AccumulatedPCM.Reset();
		EndConversation();
		return;
	}

	const FString WavPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("VirtualHuman"), TEXT("LastReply.wav"));

	if (WritePcm16ToWavFile(AccumulatedPCM, AccumulatedSampleRate, AccumulatedChannels, WavPath))
	{
		if (UAsyncActionAnimateCharacter* Action = UAsyncActionAnimateCharacter::AnimateCharacterFromWavFileAsync(
			this, Character, WavPath, FAudio2FaceEmotion(), nullptr, A2FProviderName))
		{
			Action->AudioSendCompleted.AddDynamic(this, &UVirtualHumanConversationComponent::HandleAnimationSendCompleted);
			Action->Activate();
		}
		else
		{
			OnConversationError.Broadcast(TEXT("Failed to start Audio2Face animation"));
			EndConversation();
		}
	}
	else
	{
		UE_LOG(LogVirtualHuman, Error, TEXT("Failed to write TTS audio to %s"), *WavPath);
		OnConversationError.Broadcast(TEXT("Failed to write synthesized audio for Audio2Face"));
		EndConversation();
	}

	AccumulatedPCM.Reset();
}

void UVirtualHumanConversationComponent::EndConversation()
{
	const bool bWasInProgress = bConversationInProgress;

	bConversationInProgress = false;
	bPendingEndConversation = false;
	ConsecutiveTurnCount = 0;

	SetState(EVirtualHumanState::Idle);

	if (bWasInProgress)
	{
		OnConversationEnded.Broadcast();
	}
}

void UVirtualHumanConversationComponent::SetState(EVirtualHumanState NewState)
{
	if (CurrentState == NewState)
	{
		return;
	}
	const UEnum* StateEnum = StaticEnum<EVirtualHumanState>();
	const double Now = FPlatformTime::Seconds();
	const double ElapsedInPreviousState = (LastStateChangeTime > 0.0) ? (Now - LastStateChangeTime) : 0.0;
	UE_LOG(LogVirtualHuman, Log, TEXT("State: %s -> %s (spent %.2fs in %s)"),
		*StateEnum->GetNameStringByValue((int64)CurrentState),
		*StateEnum->GetNameStringByValue((int64)NewState),
		ElapsedInPreviousState,
		*StateEnum->GetNameStringByValue((int64)CurrentState));
	LastStateChangeTime = Now;
	CurrentState = NewState;
	OnStateChanged.Broadcast(CurrentState);
}
