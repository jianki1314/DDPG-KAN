from kan import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
dataset = create_dataset(f, n_var=4, train_num=3000, device=device)

# train the model
model.fit(dataset, opt="LBFGS", steps=20, lamb=0.002, lamb_entropy=2.,save_fig=True,img_folder='KAN_picture')
model = model.prune(edge_th=1e-2)
model.plot(folder="KAN_picture")

Algorithm2:ActionInspection
Input:Proposedactionat fromDDPG-KANagent,current state  st;
Output:Final action afinal to be executed;
Receive proposed action at from the DDPG-KAN agent.;
Predict future trajectory of the ego vehicle for a prediction horizon Tpredict=1.0 second, assuming constant acceleration at.;
Predict future trajectories of surrounding vehicles for Tpredict, assuming constant velocities.;
for each surrounding vehicle within 
    Rpredict_sense = 30 meters do
        Check for potential collision with the ego vehicle’s predicted trajectory within Tpredict.;
        Collision is detected if the predicted distance between ego vehicle and surrounding vehicle becomes less than a safety distance threshold Dsafe=2.0 meters.;
        if collision detected then
            Safety Override Activated:;
            Attempt to find a safer action: set
            aoverride=min(0,at)(reduce speed or maintain current speed).;
            if aoverride still results in predicted collision(unlikely but possible in verycritical situations) then
                As a last resort, engage emergency braking:
                aoverride=amin=-3.0 m/s2.;
        return afinal=aoverride;
return afinal= at//No collision
        detected ,execute agent’s action