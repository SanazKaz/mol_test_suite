This folder contains 3 experiments with variations.

I first ran 
Reward = Lilly * 0.5 + qed * 0.5

This resulted in an excessive number of rings as well as large ring and cyclopropanes.

I then ran with 2 changes instead of 1. 

I added in ring control with this in the Lilly rules and reduced - LR (1e-5) 

dfilters = LillyDemeritsFilters(

**{"dthresh": 160,
"min_atoms": 15,
"hard_max_atoms": 50,
"max_size_rings": 7,
"min_num_rings": 1,
"min_size_rings":3 ** not actually part of lilly rules i added tis myself.
"max_num_rings": 4,
"max_size_chain": 6,
}
)

Next i added :
if three_ring_count > 0:
return -0.3

so to ban it completely. 

These results still to be properly analysed and are not included in my Viva results yet. 

3 runs:
1) lilly*0.5 + QED*0.5 (https://wandb.ai/sanazkazeminia97/Lilly_QED_lr_w0.5/runs/Lilly_QED_lr_4e-5_w0.5_sigmoid_k0.1_seed_976) 

2) Lilly with ring control plus reduced learning rate 
https://wandb.ai/sanazkazeminia97/Lilly_QED_lr_w0.5/runs/Lilly_QED_lr_1e-5_w0.5_sigmoid_k0.1_fixed_rings_seed_123 

3) Lilly with ring control, same LR but -ve conditioning for cyclopropanes.
https://wandb.ai/sanazkazeminia97/Lilly_QED_lr_w0.5/runs/Lilly_QED_lr_1e-5_w0.5_sigmoid_k0.1_fixed_rings_cyclopropane_seed_42 


All runs in: 

https://wandb.ai/sanazkazeminia97/Lilly_QED_lr_w0.5?nw=nwusersanazkazeminia97