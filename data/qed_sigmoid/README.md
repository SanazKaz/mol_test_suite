## Used QED only as a Reward

The reward was used with a sigmoid which helped seperate out middle values of QED and led to a faster climb than without it - i ran both. 

https://wandb.ai/sanazkazeminia97/qed_0-1?nw=nwusersanazkazeminia97 <- with sigmoid run till epoch 90 climbed to 0.65 then i ran it again from that epoch forward to see how high it could climb. The continued run is this https://wandb.ai/sanazkazeminia97/qed_0-1/runs/qed_0-1_sigmoid_lr_4e-5_cont_e79_seed_976 


The raw run is these runs: 
https://wandb.ai/sanazkazeminia97/qed_0-1/runs/qed_0-1_lr_4e-5_real_model_seed_42 <- sSlowler mid climb but ended up in the same place basically except seed 42.  Didnt run this for as many epochs since it didnt really need it since i already showed it could work with sigmoid. 

I generated mols with continued epoch model  (https://wandb.ai/sanazkazeminia97/qed_0-1/runs/qed_0-1_sigmoid_lr_4e-5_cont_e79_seed_976 ) and these are where our results came from.

