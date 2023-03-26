import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

index_list = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
array = [[976, 0, 0, 0, 6, 18, 0],
         [0, 997, 0, 0, 3, 0, 0],
         [1, 0, 982, 0, 0, 6, 11],
         [1, 2, 2, 955, 0, 0, 0],
         [14, 0, 0, 0, 975, 11, 0],
         [17, 0, 0, 0, 5, 978, 0],
         [0, 0, 3, 0, 0, 0, 997]]

# HW1 - Q1) 1
precision_happy = 995 / (995 + 0)
recall_happy = 995 / (995 + 5)
print('precision : ', precision_happy, '\nrecall : ', recall_happy)

# HW1 - Q1) 2
df_cm = pd.DataFrame(array, index = [i for i in index_list],
                     columns=[i for i in index_list])
print(df_cm)

sns.heatmap(df_cm, cmap="Blues")
plt.title("Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
