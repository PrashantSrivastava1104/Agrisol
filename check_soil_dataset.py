import pandas as pd

df = pd.read_csv('data/kaggle/AutoIrrigation.csv')

print("Soil Dataset Columns:")
print(df.columns.tolist())
print(f"\nTotal samples: {len(df)}")

# Show first few samples
print("\n" + "="*60)
print("SAMPLE VALUES FOR SOIL MONITORING")
print("="*60)

# Get samples with different irrigation needs
print("\n### IRRIGATION NEEDED (Pump = 1):")
irrigation_needed = df[df['Pump'] == 1].iloc[0]
print(f"Temperature: {irrigation_needed['Temperature']:.1f}")
print(f"Humidity: {irrigation_needed['Humidity']:.1f}")
print(f"pH: {irrigation_needed['pH']:.1f}")
print(f"EC: {irrigation_needed['EC']:.1f}")
print(f"N: {irrigation_needed['Nitrogen']:.1f}")
print(f"P: {irrigation_needed['Phosphorus']:.1f}")
print(f"K: {irrigation_needed['Potassium']:.1f}")
print(f"Moisture: {irrigation_needed['Soil_Moisture']:.1f}")

print("\n### NO IRRIGATION NEEDED (Pump = 0):")
no_irrigation = df[df['Pump'] == 0].iloc[0]
print(f"Temperature: {no_irrigation['Temperature']:.1f}")
print(f"Humidity: {no_irrigation['Humidity']:.1f}")
print(f"pH: {no_irrigation['pH']:.1f}")
print(f"EC: {no_irrigation['EC']:.1f}")
print(f"N: {no_irrigation['Nitrogen']:.1f}")
print(f"P: {no_irrigation['Phosphorus']:.1f}")
print(f"K: {no_irrigation['Potassium']:.1f}")
print(f"Moisture: {no_irrigation['Soil_Moisture']:.1f}")

# Get a few more samples
print("\n### ADDITIONAL TEST CASES:")
for i in range(3):
    sample = df.iloc[i*100]
    pump_status = "IRRIGATION NEEDED" if sample['Pump'] == 1 else "NO IRRIGATION"
    print(f"\nTest Case {i+1} ({pump_status}):")
    print(f"  Temp={sample['Temperature']:.1f}, Humidity={sample['Humidity']:.1f}, pH={sample['pH']:.1f}")
    print(f"  EC={sample['EC']:.1f}, N={sample['Nitrogen']:.1f}, P={sample['Phosphorus']:.1f}, K={sample['Potassium']:.1f}")
    print(f"  Moisture={sample['Soil_Moisture']:.1f}")
