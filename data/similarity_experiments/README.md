These experiments are modelled after the first reinvent paper:

I tried fingerprint similarity to aripiprazole from the guacamol paper: these didnt work - the wandb plots are from one of the runs : data/similarity_experiments/ecfp_fcfp_sim/wandb_ecfp_fcfp_csv 

https://wandb.ai/sanazkazeminia97/aripiprazole_sim?nw=nwusersanazkazeminia97 <- all sim runs can be found here if needed.

the substructure runs refer to the run with dichlorobenzene which also didnt work very well. The reason for this is because fingerprints are inherently binary with presence or absence despite the score being between 0-1. This means the model, which is data hungry does not have enough data for gradient ascent toward something more similar.

The same thing was observed with FCFP4 fingerprint length 2048. 


Next i tried it with SuCOS (normalised) and we saw a small climb over a long period of time - find the results here:
 https://wandb.ai/sanazkazeminia97/SuCOS_0-1?nw=nwusersanazkazeminia97 

https://wandb.ai/sanazkazeminia97/SuCOS_0-1/runs/SuCOS_lr_1e-5_clip_0.1_centered_rewards_seed_42 

These climbined slowly - but look at raw score mean and not reward mean. Here i had to Exponential moving average with 0 centering every batch of mols to provide any signal. Without this it didnt work. But also i changed 2 things at once - the lr was 4e-5 which was too high and caused clip frac and kl spikes - might have worked if i just lowered the learning rate. I lowered it 1e-4

