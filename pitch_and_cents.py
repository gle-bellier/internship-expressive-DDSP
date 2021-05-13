from os import get_blocking
import torch
import librosa as li 


pitch_size = 100
max_freq = li.midi_to_hz(pitch_size-1)

print(max_freq)

freq = torch.abs((torch.randn(4, 20, 1) *10 + 10)) * 10
freq = torch.clip(freq, max = max_freq)


print("FREQUENCIES")
print(freq)
midi_pitch = torch.tensor(li.hz_to_midi(freq))
midi_pitch = torch.round(midi_pitch).long()

print("MIDI FREQ")
print(midi_pitch.size())


round_freq = torch.tensor(li.midi_to_hz(midi_pitch))
print("ROUND FREQUENCIES")
print(round_freq.size())

cents = (1200 * torch.log2(freq / round_freq)).long()
print("CENTS")
print(cents.size())

pitch_array = torch.zeros(freq.size(0), freq.size(1), pitch_size)

for i in range(0, pitch_array.size(0)):
    for j in range(0, pitch_array.size(1)):
        pitch_array[i, j, midi_pitch[i, j, 0]] = 1

cents_array = torch.zeros(freq.size(0), freq.size(1), 100)

for i in range(0, pitch_array.size(0)):
    for j in range(0, pitch_array.size(1)):
        pitch_array[i, j, cents[i, j, 0] + 50] = 1




gen_pitch = torch.argmax(pitch_array, dim = -1)
gen_cents = torch.argmax(cents_array, dim = -1) - 50

print("gen pitch", gen_pitch.shape)
gen_freq = torch.tensor(li.midi_to_hz(gen_pitch)) * torch.pow(2, gen_cents/1200)
gen_freq = gen_freq.view(gen_freq.size(0), gen_freq.size(1), 1)

print("GEN FREQUENCIES")
print(gen_freq)



print("DIFF")
print((freq - gen_freq)/freq)
