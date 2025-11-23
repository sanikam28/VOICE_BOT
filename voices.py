from elevenlabs import ElevenLabs

client = ElevenLabs(api_key="sk_8ee78b1745141096071e40d847b378d9fc189ac7292aa820")

voices = client.voices.get_all()
for v in voices.voices:
    print(v.voice_id, v.name)
