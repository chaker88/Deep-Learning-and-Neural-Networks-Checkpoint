import queue
import pydub
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
from sample_utils.turn import get_ice_servers


def main():
    recognizer = sr.Recognizer()

    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration={
            "iceServers": get_ice_servers()
        },
        media_stream_constraints={
            "audio": True,
        },
    )

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    # Corrected channel identification
                    channels=1 if audio_frame.layout.name == 'mono' else 2,
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                st.session_state["audio_buffer"] += sound_chunk
        else:
            status_indicator.write("AudioReceiver is not set. Abort.")
            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        st.info("Performing speech recognition with Sphinx...")
        raw_audio_data = audio_buffer.raw_data
        sample_width = audio_buffer.sample_width
        frame_rate = audio_buffer.frame_rate

        audio_data = sr.AudioData(
            raw_audio_data,
            sample_rate=frame_rate,
            sample_width=sample_width,
        )

        try:
            text = recognizer.recognize_sphinx(
                audio_data, language='en-US')  # Using Sphinx recognizer
            st.success("Transcription: " + text)
        except sr.UnknownValueError:
            st.warning("Could not understand the audio")
        except sr.RequestError as e:
            st.error(f"Error: {e}")

        st.info("Writing WAV to disk")
        audio_buffer.export("speech_recognition/temp.wav", format="wav")

        # Reset
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()


if __name__ == "__main__":
    main()
