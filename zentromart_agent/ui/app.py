import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent.session import ConversationSession
from src.agent.orchestrator import ZentroMartAgent
from src.voice.stt import transcribe
from src.voice.tts import synthesize
from src.voice.audio_utils import clean_for_tts

# Global agents
text_agent = ZentroMartAgent(mode="full")
voice_agent = ZentroMartAgent(mode="full")

# Global sessions (per session_id or single user demo)
text_session = ConversationSession(session_id="ui_text_sess", channel="text")
voice_session = ConversationSession(session_id="ui_voice_sess", channel="voice")

def chat_respond(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        return history, "", text_session.state, text_session.detected_lang, list(text_session.section_ids_used)

    try:
        reply = text_agent.respond(user_message, text_session)
    except Exception as e:
        reply = f"I encountered an error processing your inquiry: {e}"

    # Support Gradio 5/6 messages format and older tuple format
    if history and isinstance(history[0], (list, tuple)):
        history.append((user_message, reply))
    else:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

    status = text_session.state
    lang = text_session.detected_lang
    sources = list(text_session.section_ids_used)

    return history, "", status, lang, sources

def clear_text_chat():
    global text_session
    text_session = ConversationSession(session_id="ui_text_sess", channel="text")
    return [], "GREETING", "en", []

def voice_turn(audio_file_path, language_code):
    if not audio_file_path:
        return "Please record or upload audio.", "", None, "GREETING", []

    # 1. Transcribe
    try:
        stt_res = transcribe(audio_file_path, language_code=language_code)
        transcript = stt_res.get("transcript")
    except Exception as e:
        transcript = None

    if not transcript or not transcript.strip():
        reply = "Sorry, I didn't catch that. Could you say it again?"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_wav = tf.name
        try:
            synthesize(reply, "en", out_wav)
        except Exception:
            out_wav = None
        return "", reply, out_wav, voice_session.state, []

    # 2. Agent response
    reply = voice_agent.respond(transcript, voice_session, lang_hint=language_code)

    # 3. TTS synthesis
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_wav = tf.name
    tts_text = clean_for_tts(reply, voice_session.detected_lang)
    try:
        synthesize(tts_text, voice_session.detected_lang, out_wav)
    except Exception as e:
        print(f"TTS synthesis error: {e}")
        out_wav = None

    return (
        transcript,
        reply,
        out_wav,
        voice_session.state,
        list(voice_session.section_ids_used)
    )

def clear_voice_chat():
    global voice_session
    voice_session = ConversationSession(session_id="ui_voice_sess", channel="voice")
    return "", "", None, "GREETING", []

def create_ui():
    import gradio as gr

    with gr.Blocks(title="ZentroMart VoiceOps AI Support") as demo:
        gr.Markdown(
            """
            # 🛒 ZentroMart VoiceOps AI Support Assistant
            ### Grounded Customer Support with Gnani Speech Models & RAG
            *Fixed Date: 20 October 2026 | Powered by Google Gemini & Gnani.ai*
            """
        )

        with gr.Tabs():
            # Tab 1: Text Chat
            with gr.TabItem("💬 Text Chat Support"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=480, label="Support Conversation")
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Type your inquiry (e.g. 'My order 61698151 still hasn't arrived. Can I cancel it?')",
                                label="Customer Message",
                                scale=8
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=2)
                        clear_btn = gr.Button("Reset Conversation", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Session Telemetry")
                        state_display = gr.Textbox(label="Verification State", value="GREETING")
                        lang_display = gr.Textbox(label="Detected Language", value="en")
                        sources_display = gr.JSON(label="Retrieved Policy Sections", value=[])

                send_btn.click(
                    chat_respond,
                    inputs=[msg_box, chatbot],
                    outputs=[chatbot, msg_box, state_display, lang_display, sources_display]
                )
                msg_box.submit(
                    chat_respond,
                    inputs=[msg_box, chatbot],
                    outputs=[chatbot, msg_box, state_display, lang_display, sources_display]
                )
                clear_btn.click(
                    clear_text_chat,
                    outputs=[chatbot, state_display, lang_display, sources_display]
                )

            # Tab 2: Voice Support
            with gr.TabItem("🎙️ Voice Support (Gnani STT + TTS)"):
                gr.Markdown("Speak to the agent using Gnani Speech Models (English, Hindi, Tamil, Telugu, etc.)")
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Customer Audio")
                        lang_select = gr.Dropdown(
                            choices=[
                                "en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN",
                                "bn-IN", "gu-IN", "mr-IN", "pa-IN", "ml-IN"
                            ],
                            value="en-IN",
                            label="Gnani STT Language Code"
                        )
                        voice_btn = gr.Button("Submit Audio", variant="primary")
                        clear_voice_btn = gr.Button("Reset Voice Session", variant="secondary")

                    with gr.Column(scale=3):
                        stt_out = gr.Textbox(label="Transcribed Text (Gnani STT)")
                        reply_out = gr.Textbox(label="Agent Text Reply")
                        audio_out = gr.Audio(label="Spoken Reply (Gnani TTS)", autoplay=True)
                        voice_state = gr.Textbox(label="Verification State", value="GREETING")
                        voice_sources = gr.JSON(label="Retrieved Sources", value=[])

                voice_btn.click(
                    voice_turn,
                    inputs=[audio_in, lang_select],
                    outputs=[stt_out, reply_out, audio_out, voice_state, voice_sources]
                )
                clear_voice_btn.click(
                    clear_voice_chat,
                    outputs=[stt_out, reply_out, audio_out, voice_state, voice_sources]
                )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    import gradio as gr
    demo.launch(server_name="localhost", server_port=7860, theme=gr.themes.Soft(), share=False)
