using Evie
using GLMakie
using DataStructures: CircularBuffer

# start Evie
fs, _buf, _ = initFsBuf()  # initialize the audio stream
audio_obs = Observable(_buf)
audio_buf = CircularBuffer{Float32}(1024*52) # ≈ 1.109s, circular buffer for the audio
speech_obs = Observable("[ Silence ]") # text input
btn_label = Observable("⬤") # button label
plotSpectrogram(audio_obs, fs, speech_obs, btn_label)
# start listening to the microphone
t_seconds = 20 # seconds
btn_label[] = "⫷"
model_att=joinpath(@__DIR__, "models/gguf/whisper-1b-english.Q4_K_S.gguf")

# start 
listenEvie(audio_obs, speech_obs, audio_buf, t_seconds, model_att;
    transcribe_text=true)


# connect to Llama2
using Llama2
llama_model = joinpath(@__DIR__, "models/llama-2-7b-chat.Q4_K_S.gguf")
model_llama = load_gguf_model(llama_model);

sample(model_llama, "how to sum two numbers in Julia?"; temperature=0.7f0) # txt_obs[]

output_prompt=[]
sampleObs(model_llama, "What is love?", output_prompt; temperature=0.7f0) # txt_obs[]

# on(button.clicks) do _
#     if playing[]
#         button.label[] = "⬤"
#         playing[] = false
#     else
#         button.label[] = "⫷"
#         playing[] = true
#         listenEvie(t_seconds, buf_obs, txt_query, circ_buf, model_att)
#     end
# end