import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('project4.csv')

X = data[['Danceability', 'Energy', 'Key']]

k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
data['Cluster'] = kmeans.labels_
plt.scatter(data['Danceability'], data['Energy'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('Clustered Data')
plt.show()


#####################################################################################


# Select the first 10 rows
first_10_rows = data.head(10)

# Calculate the mean values for 'Danceability' and 'Energy' columns
danceability_mean = first_10_rows['Danceability'].mean()
energy_mean = first_10_rows['Energy'].mean()

# Pie chart
labels = ['Danceability', 'Energy']
sizes = [danceability_mean, energy_mean]
colors = ['lightblue', 'lightgreen']
explode = (0.1, 0)  # explode the 1st slice (optional)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage distribution of Danceability and Energy')
plt.show()
