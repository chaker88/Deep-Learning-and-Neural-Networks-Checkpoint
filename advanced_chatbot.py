from speech_reco.deepgarm_speech_recognition import transcribe_audio_chunks,extract_transcript_confidence
from chat_bot.chat_bot import chatbot
import json
import queue
import pydub
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode


DEEPGRAM_API_KEY ='592375b5d564169a2040dbdf4a064ae9d18fbf05'   

def main():
    st.title("Welcome to my advanced chat bot")
    st.sidebar.title("Input Options")
    input_choice = st.sidebar.radio("Select Input Method:", ("Keyboard", "Voice Recognition"))

    # Using webrtc for voice input
    if input_choice == "Voice Recognition":
        webrtc_ctx = webrtc_streamer(
            key="sendonly-audio",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=256,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
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
                        channels=1 if audio_frame.layout.name == 'mono' else 2,
                    )
                    sound_chunk += sound
                if len(sound_chunk) > 0:
                    st.session_state["audio_buffer"] += sound_chunk
            else:
                status_indicator.write("AudioReceiver stop.")
                break

        audio_buffer = st.session_state["audio_buffer"]

        if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
            st.info("Performing speech recognition with Deepgram...")
            # Perform transcription and display results
            transcription = transcribe_audio_chunks(audio_buffer, DEEPGRAM_API_KEY, "en")
            if transcription:
                formatted_json = json.dumps(transcription.to_dict(), indent=4)
                transc, confi = extract_transcript_confidence(formatted_json)
                st.success("Transcription success")
                st.write('Transcription:', transc)
                st.write('Confidence:', confi)
                response = chatbot(transc)  # Replace with your chatbot function
                st.write("Chatbot reponse :", response)  # Display chatbot response
            else:
                st.info("Writing WAV to disk")

            audio_buffer.export("Deep_Learning_and_Neural_Networks_Checkpoint/temp.wav", format="wav")
            st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
            st.session_state['language_to_transcript'] = ""

    # Using keyboard input for chatbot
    else:  # Keyboard input selected
        user_input = st.text_input("Enter your message:")
        if st.button("Send"):
            response = chatbot(user_input)  # Replace with your chatbot function
            st.write("Chatbot reponse :", response)  # Display chatbot response

if __name__ == "__main__":
    main()

