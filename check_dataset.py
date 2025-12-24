import pandas as pd

df = pd.read_csv('data/kaggle/Crop_recommendation.csv')

print("Available crops in dataset:")
print(df['label'].unique())
print(f"\nTotal samples: {len(df)}")
print(f"\nSamples per crop:")
print(df['label'].value_counts())

print("\n" + "="*60)
print("SAMPLE VALUES FROM DATASET (First sample of each crop)")
print("="*60)

for crop in sorted(df['label'].unique()):
    sample = df[df['label']==crop].iloc[0]
    print(f'\n{crop.upper()}:')
    print(f'  N={int(sample.N)}, P={int(sample.P)}, K={int(sample.K)}')
    print(f'  Temperature={sample.temperature:.1f}, Humidity={sample.humidity:.1f}')
    print(f'  pH={sample.ph:.1f}, Rainfall={sample.rainfall:.1f}')
