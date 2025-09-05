import pandas as pd
import matplotlib.pyplot as plt


file_path ="C:\\Users\\HP\\OneDrive\\Desktop\\Skillcraft_Internship\\API_SP.POP.TOTL_DS2_en_csv_v2_564351.csv"
df = pd.read_csv(file_path, skiprows=4)


year = "2020"


pop_data = df[["Country Name", year]].dropna()


top10 = pop_data.sort_values(by=year, ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(top10["Country Name"], top10[year])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Most Populous Countries in 2020")
plt.xlabel("Country")
plt.ylabel("Population")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
plt.hist(pop_data[year]/1e6, bins=30, edgecolor="black")  
plt.title("Distribution of Country Populations (2020)")
plt.xlabel("Population (millions)")
plt.ylabel("Number of Countries")
plt.tight_layout()
plt.show()
