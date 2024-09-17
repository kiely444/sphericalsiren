import pyshtools as pysh
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from inr import SphericalSiren
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.gridspec as gridspec

topo_coeffs = pysh.datasets.Earth.Earth2014.tbi(lmax=60) / 1000.0
world_map_array = topo_coeffs.expand(lmax=60).to_array()

world_map_min = world_map_array.min()
world_map_max = world_map_array.max()
world_map_array = 2 * (world_map_array - world_map_min) / (world_map_max - world_map_min) - 1

world_map_array = np.flipud(world_map_array)

fig = plt.figure(figsize=(12, 10))
ax_signal = fig.add_subplot(111, projection=ccrs.Mollweide())
lon = np.linspace(-180, 180, world_map_array.shape[1])
lat = np.linspace(-90, 90, world_map_array.shape[0])
lon, lat = np.meshgrid(lon, lat)

im0 = ax_signal.pcolormesh(lon, lat, world_map_array, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
ax_signal.set_title('Ground Truth')
fig.colorbar(im0, ax=ax_signal, orientation='horizontal', pad=0.05, aspect=50)

plt.tight_layout()
plt.savefig('images/world_map.png')

n_latitudes, n_longitudes = world_map_array.shape
theta = torch.linspace(0, np.pi, n_latitudes)
phi = torch.linspace(0, 2 * np.pi, n_longitudes)
theta_grid, phi_grid = torch.meshgrid(theta, phi)
theta_phi_tensor = torch.stack((theta_grid.flatten(), phi_grid.flatten()), dim=-1)
target_signal_tensor = torch.tensor(world_map_array.flatten(), dtype=torch.float64)

num_epochs = 50
NUMLAYERS = 1
neurons=50
lmax=12

#g(x) = 1-x^2/2
siren = SphericalSiren(lmax=lmax, hidden_layers=NUMLAYERS, neurons=neurons, useSine=False)
optimizer = optim.Adam(siren.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    siren.train()
    optimizer.zero_grad()

    outputs = siren(theta_phi_tensor).squeeze()
    loss = nn.MSELoss()(outputs, target_signal_tensor)
    loss.backward()
    optimizer.step()

    #if (epoch + 1) % 100 == 0 or epoch==0:
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}') #SAVE THIS

siren.eval()
with torch.no_grad():
    predicted_signal = siren(theta_phi_tensor)
predicted_signal_np = predicted_signal.numpy().reshape((n_latitudes, n_longitudes))
snr = 20 * np.log10(np.linalg.norm(target_signal_tensor.unsqueeze(1))/np.linalg.norm(target_signal_tensor.unsqueeze(1)-predicted_signal)) 
print(f'SNR: {snr}')
fig = plt.figure(figsize=(12, 10))

ax_signal = fig.add_subplot(111, projection=ccrs.Mollweide())
lon = np.linspace(-180, 180, predicted_signal_np.shape[1])
lat = np.linspace(-90, 90, predicted_signal_np.shape[0])
lon, lat = np.meshgrid(lon, lat)
im0 = ax_signal.pcolormesh(lon, lat, predicted_signal_np, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
ax_signal.set_title(f'1-x^2/2 World Map Recreation - SNR = {snr}')
fig.colorbar(im0, ax=ax_signal, orientation='horizontal', pad=0.05, aspect=50)

plt.tight_layout()
plt.savefig('worldmapimages/xsquaredworldmap.png')

#g(x) = sine
siren = SphericalSiren(lmax=lmax, hidden_layers=NUMLAYERS, neurons=neurons, useSine=True)
optimizer = optim.Adam(siren.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    siren.train()
    optimizer.zero_grad()

    outputs = siren(theta_phi_tensor).squeeze()
    loss = nn.MSELoss()(outputs, target_signal_tensor)
    loss.backward()
    optimizer.step()

    #if (epoch + 1) % 100 == 0 or epoch==0:
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}') #SAVE THIS"""

siren.eval()
with torch.no_grad():
    predicted_signal = siren(theta_phi_tensor)
predicted_signal_np = predicted_signal.numpy().reshape((n_latitudes, n_longitudes))
snr = 20 * np.log10(np.linalg.norm(target_signal_tensor.unsqueeze(1))/np.linalg.norm(target_signal_tensor.unsqueeze(1)-predicted_signal)) 
print(f'SNR: {snr}')
fig = plt.figure(figsize=(12, 10))

ax_signal = fig.add_subplot(111, projection=ccrs.Mollweide())
lon = np.linspace(-180, 180, predicted_signal_np.shape[1])
lat = np.linspace(-90, 90, predicted_signal_np.shape[0])
lon, lat = np.meshgrid(lon, lat)
im0 = ax_signal.pcolormesh(lon, lat, predicted_signal_np, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
ax_signal.set_title(f'Sine World Map Recreation - SNR = {snr}')
fig.colorbar(im0, ax=ax_signal, orientation='horizontal', pad=0.05, aspect=50)

plt.tight_layout()
plt.savefig('worldmapimages/sineworldmap.png')
