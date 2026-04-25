# %%
print("hello from kagglet!")

# %%
total = sum(i * i for i in range(1_000_000))
print(f"sum(i^2 for i in [0, 1e6)) = {total}")
