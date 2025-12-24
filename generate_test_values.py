import pandas as pd

df = pd.read_csv('data/kaggle/Crop_recommendation.csv')

output = []
output.append("# AgriSol Dataset - Accurate Test Values\n")
output.append(f"Total samples: {len(df)}\n")
output.append(f"Available crops: {len(df['label'].unique())}\n\n")

output.append("## Crops in Dataset:\n")
for crop in sorted(df['label'].unique()):
    count = len(df[df['label']==crop])
    output.append(f"- {crop}: {count} samples\n")

output.append("\n## Sample Values (First example from dataset for each crop):\n\n")

for crop in sorted(df['label'].unique()):
    sample = df[df['label']==crop].iloc[0]
    output.append(f"### {crop.upper()}\n")
    output.append(f"- **Nitrogen (N):** {int(sample.N)}\n")
    output.append(f"- **Phosphorus (P):** {int(sample.P)}\n")
    output.append(f"- **Potassium (K):** {int(sample.K)}\n")
    output.append(f"- **Temperature:** {sample.temperature:.1f}°C\n")
    output.append(f"- **Humidity:** {sample.humidity:.1f}%\n")
    output.append(f"- **pH Level:** {sample.ph:.1f}\n")
    output.append(f"- **Rainfall:** {sample.rainfall:.1f}mm\n\n")

with open('test_values_guide.md', 'w') as f:
    f.writelines(output)

print("Test values guide created: test_values_guide.md")
