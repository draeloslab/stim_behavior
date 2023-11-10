import deeplabcut, dlclive
import pandas as pd
import matplotlib.pyplot as plt
dlclive.benchmark_videos('/home/jakejoseph/Desktop/FES_V1-Joseph-2023-10-16/exported-models/DLC_FES_V1_resnet_50_iteration-0_shuffle-1',
                          '/home/jakejoseph/Desktop/FES_V1-Joseph-2023-10-16/videos/NapierCharacterization20230801NoText.mp4',
                          resize=0.5, pcutoff=0.9, display_radius=8, cmap='bmy', output= '/home/jakejoseph/Desktop/FES_V1-Joseph-2023-10-16/')


filepath = '/home/jakejoseph/Desktop/FES_V1-Joseph-2023-10-16/benchmark_eros_CPU_2.pickle'

df = pd.read_pickle(filepath)
# print(df['inference_times'])

plt.plot(1/df['inference_times'][0])
plt.xlabel('Frame')
plt.ylabel('Frames Per Second')
plt.title('Inference Time NHP Hand Posture (CPU)')
#plt.ylim([0,0.2])
plt.show()