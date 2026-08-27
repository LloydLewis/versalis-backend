#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "VirtualHumanConversationComponent.generated.h"

class UACEASRSubsystem;
class UACETTSSubsystem;

UENUM(BlueprintType)
enum class EVirtualHumanState : uint8
{
	Idle,
	Listening,
	Thinking,
	Speaking
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FVirtualHumanStateChanged, EVirtualHumanState, NewState);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FVirtualHumanTextEvent, const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FVirtualHumanErrorEvent, const FString&, ErrorMessage);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FVirtualHumanConversationEvent);

/**
 * Owns the full conversation loop for one virtual human: ACE ASR capture -> your
 * OpenAI-compatible LLM server (aisegment.py behind FastAPI) -> ACE TTS synthesis ->
 * Audio2Face lipsync, driven entirely through the ACE ASR / ACE TTS GameInstance
 * subsystems (no ACEASRComponent / ACETTSComponent needed on the actor).
 *
 * Add this to the NPC actor that owns the ACEAudioCurveSourceComponent (e.g. BP_Oskar).
 * This is no longer push-to-talk: a single StartListening() call (e.g. one keypress)
 * begins the conversation, and the component keeps looping listen -> LLM -> speak turns
 * on its own (each utterance auto-finalizes via ACE ASR's inactivity watchdog - see
 * MaxCaptureInactivitySeconds in Project Settings) until the LLM's reply signals the
 * conversation is over. The LLM signals this by including the EndConversationMarker
 * text (see the .cpp) anywhere in its reply; aisegment.py/server.py is responsible for
 * deciding when to emit it. Only StartListening() needs to be wired from input.
 */
UCLASS(ClassGroup = (VirtualHuman), meta = (BlueprintSpawnableComponent))
class KAIROSSAMPLE_API UVirtualHumanConversationComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UVirtualHumanConversationComponent();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	/** Base URL of your OpenAI-compatible LLM server, e.g. http://127.0.0.1:8008/v1 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human|LLM")
	FString LLMBaseUrl = TEXT("http://127.0.0.1:8008/v1");

	/** Sent as the "model" field. Your FastAPI server can ignore this. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human|LLM")
	FString LLMModelId = TEXT("local");

	/** Bearer token; leave empty for a local, unauthenticated server. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human|LLM")
	FString LLMApiKey;

	/**
	 * Actor to animate via Audio2Face (must already have a working
	 * ACEAudioCurveSourceComponent, per your existing lipsync setup).
	 * Leave unset to default to the owning actor.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human|Lipsync")
	TObjectPtr<AActor> LipsyncCharacterOverride = nullptr;

	/**
	 * Other MetaHuman actors that can also deliver the reply. Each one needs its own
	 * working ACEAudioCurveSourceComponent lipsync setup, exactly like the primary
	 * character (LipsyncCharacterOverride, or this component's owner if that's unset).
	 * Every time a reply is ready to be spoken, one actor is picked at random out of
	 * {primary character + this list} - drop every extra MetaHuman you want in the
	 * random pool here. Only one conversation component should exist in the level
	 * (ASR/TTS are shared GameInstance subsystems), so this is how you get more than
	 * one MetaHuman without duplicating the component.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human|Lipsync")
	TArray<TObjectPtr<AActor>> AdditionalVirtualHumans;

	/** Matches the A2FProviderName used by your existing AnimateCharacterFromWavFileAsync setup ("Default" unless you registered a named provider). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human|Lipsync")
	FName A2FProviderName = FName("Default");

	/**
	 * Microphone device index to use, from GetACEASRInputDevices. -1 (default) means
	 * "auto-pick the first device the plugin enumerates" instead of asking Windows for
	 * its default capture device - WASAPI's default-device role is easy to leave unset
	 * on Bluetooth/USB headsets after a reconnect, which makes StartMicrophoneTranscription
	 * fail with "no default capture device found" even though the device itself is fine.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human|ASR")
	int32 PreferredCaptureDeviceIndex = -1;

	/**
	 * Optional runaway-loop safety backstop: if the LLM never emits the end-conversation
	 * marker, force the conversation to end after this many completed turns. 0 (default)
	 * disables this and leaves ending the conversation entirely up to the LLM.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Virtual Human", meta = (ClampMin = "0"))
	int32 MaxConsecutiveTurns = 0;

	/**
	 * Call to begin a conversation (e.g. on a single keypress - "T" in the setup guide).
	 * Starts listening for the first turn; once the user pauses, ASR auto-finalizes the
	 * transcript (see MaxCaptureInactivitySeconds) and the component runs the LLM ->
	 * TTS -> lipsync turn, then automatically starts listening again for the next turn.
	 * This repeats on its own - no further input is needed - until the LLM's reply signals
	 * the conversation should end. Ignored while a turn is already in progress (Listening/
	 * Thinking/Speaking) or while a conversation is already active and idly waiting on the
	 * next pause; it's also what the component calls internally to resume listening after
	 * each turn.
	 */
	UFUNCTION(BlueprintCallable, Category = "Virtual Human")
	void StartListening();

	/** Manually cuts the current utterance short and finalizes it early. Not required for normal use - ASR auto-finalizes on a pause. */
	UFUNCTION(BlueprintCallable, Category = "Virtual Human")
	void StopListening();

	UFUNCTION(BlueprintPure, Category = "Virtual Human")
	EVirtualHumanState GetState() const { return CurrentState; }

	/** True from the moment StartListening() begins a new conversation until the LLM ends it (or an error aborts it). */
	UFUNCTION(BlueprintPure, Category = "Virtual Human")
	bool IsConversationInProgress() const { return bConversationInProgress; }

	/** Convenience getter so player input Blueprints don't need to search the level for this component. Returns the most recently initialized instance. */
	UFUNCTION(BlueprintCallable, Category = "Virtual Human", meta = (WorldContext = "WorldContextObject"))
	static UVirtualHumanConversationComponent* GetActiveVirtualHuman(const UObject* WorldContextObject);

	UPROPERTY(BlueprintAssignable, Category = "Virtual Human|Events")
	FVirtualHumanStateChanged OnStateChanged;

	/** Final user transcript, for optional subtitle UI. */
	UPROPERTY(BlueprintAssignable, Category = "Virtual Human|Events")
	FVirtualHumanTextEvent OnUserTranscript;

	/** LLM reply text, for optional subtitle UI. Fires before TTS playback starts. */
	UPROPERTY(BlueprintAssignable, Category = "Virtual Human|Events")
	FVirtualHumanTextEvent OnAssistantReply;

	UPROPERTY(BlueprintAssignable, Category = "Virtual Human|Events")
	FVirtualHumanErrorEvent OnConversationError;

	/** Fires once when StartListening() begins a brand-new conversation (not on each internal per-turn re-listen). */
	UPROPERTY(BlueprintAssignable, Category = "Virtual Human|Events")
	FVirtualHumanConversationEvent OnConversationStarted;

	/** Fires once the conversation is over - either the LLM decided to end it, an error aborted it, or MaxConsecutiveTurns was hit. */
	UPROPERTY(BlueprintAssignable, Category = "Virtual Human|Events")
	FVirtualHumanConversationEvent OnConversationEnded;

private:
	UFUNCTION()
	void HandleASRReady();

	UFUNCTION()
	void HandleASRError(const FString& ErrorMessage);

	UFUNCTION()
	void HandleTranscriptFinalized(const FString& FinalTranscript);

	UFUNCTION()
	void HandleTTSReady();

	UFUNCTION()
	void HandleTTSError(const FString& ErrorMessage);

	UFUNCTION()
	void HandleSpeechFinished(const FString& Text, bool bSucceeded);

	UFUNCTION()
	void HandleAudioSamplesReady(TArray<int16> Samples, int32 SampleRate, int32 Channels, bool bIsFinalChunk);

	/**
	 * Fires when Audio2Face finishes receiving the WAV (AsyncActionAnimateCharacter's
	 * AudioSendCompleted). This - not ACE TTS's own OnSpeechFinished - is what returns
	 * the state machine to Idle: TTS's own AudioComponent is volume-muted (see
	 * DefaultGame.ini), and UE's audio engine can virtualize/skip completion callbacks
	 * for sounds it judges inaudible, so OnSpeechFinished can never fire.
	 */
	UFUNCTION()
	void HandleAnimationSendCompleted(bool bSuccess);

	UFUNCTION()
	void HandleLLMResponse(const FString& Content);

	UFUNCTION()
	void HandleLLMError(const FString& ErrorMessage);

	void SetState(EVirtualHumanState NewState);
	AActor* ResolveLipsyncCharacter() const;
	void FlushAccumulatedAudioToLipsync();

	/** Resets conversation-tracking state, sets Idle, and broadcasts OnConversationEnded (only if a conversation was actually in progress). */
	void EndConversation();

	UPROPERTY(Transient)
	TObjectPtr<UACEASRSubsystem> ASRSubsystem;

	UPROPERTY(Transient)
	TObjectPtr<UACETTSSubsystem> TTSSubsystem;

	EVirtualHumanState CurrentState = EVirtualHumanState::Idle;
	double LastStateChangeTime = 0.0;

	TArray<int16> AccumulatedPCM;
	int32 AccumulatedSampleRate = 0;
	int32 AccumulatedChannels = 0;

	/** True from StartListening()'s first turn until the conversation ends. */
	bool bConversationInProgress = false;

	/** Set by HandleLLMResponse when the reply carried the end-conversation marker; consumed once the reply finishes playing. */
	bool bPendingEndConversation = false;

	/** Turns completed in the current conversation; reset by EndConversation(). Compared against MaxConsecutiveTurns. */
	int32 ConsecutiveTurnCount = 0;
};
