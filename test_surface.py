from vol_surface.surface import build_surface, plot_surface
df = build_surface("AAPL")

print(f"Got {len(df)} contracts")
print(df.head())

plot_surface(df)