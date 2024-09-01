import os
import hmac
import tempfile

import streamlit

from streamlit_mic_recorder import mic_recorder

import replicate

import azure.cognitiveservices.speech as speechsdk


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(
            streamlit.session_state["password"], streamlit.secrets["password"]
        ):
            streamlit.session_state["password_correct"] = True
            del streamlit.session_state["password"]  # Don't store the password.
        else:
            streamlit.session_state["password_correct"] = False

    # Return True if the password is validated.
    if streamlit.session_state.get("password_correct", False):
        return True

    # Show input for password.
    streamlit.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in streamlit.session_state:
        streamlit.error("😕 Password incorrect")
    return False


if not check_password():
    streamlit.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
streamlit.title("LLAMA 2 Chat")
streamlit.header("Created By Shirit Eshed")
streamlit.markdown(
"""
1. Click start recording to start recording. Record a question for the LLM model
2. Click stop recordign to send the recording to the model
3. The answer should appear in text and in audio
"""
)


def llama(prompt):
    output = replicate.run("meta/llama-2-70b-chat", input={"prompt": prompt})
    return "".join(output)


audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    format="wav",
    callback=None,
    args=(),
    kwargs={},
    key=None,
)

human = streamlit.chat_message(name="human")
ai = streamlit.chat_message(name="ai")

if audio:
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ["AZURE_SPEECH_SUBSCRIPTION"], region=os.environ["AZURE_SPEECH_REGION"]
    )
    speech_config.speech_recognition_language = "en-US"
    with tempfile.NamedTemporaryFile("wb", delete=False) as audio_file:
        audio_file.write(audio["bytes"])
    audio_input_config = speechsdk.audio.AudioConfig(filename=audio_file.name)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_input_config
    )
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcription = speech_recognition_result.text
        human.write(transcription)
        response = llama(transcription)
        output_file = tempfile.NamedTemporaryFile("w", delete=False)
        audio_output_config = speechsdk.audio.AudioOutputConfig(
            use_default_speaker=True,
            filename=output_file.name,
        )
        speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_output_config
        )
        speech_synthesis_result = speech_synthesizer.speak_text_async(response).get()
        if (
            speech_synthesis_result.reason
            == speechsdk.ResultReason.SynthesizingAudioCompleted
        ):
            streamlit.audio("output.wav")
            ai.write(response)
        else:
            streamlit.write("error occured")
    else:
        streamlit.write("Could not detect speech or error occured")

