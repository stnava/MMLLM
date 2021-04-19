# automated landmarking training and application

there are two methods - one for reorientation ( finding rotations ) and
another for automated landmarking.

both networks use a fixed template space - template* in this data directory.
the template is inconsequential, theoretically, but does impact the details 
of the weights as it provides the specific reference frame for rotation 
and for the overall physical space.  a different template could easily 
be used but would necessitate retraining or a potentially complex 
transformation of the network weights.

## reorientation

training code - `orient_mouse_3D.R`

 the inference code is shown in the file `auto_lm_mouse_ct_inference.R` 

1. an efficient resnet regression of rotation parameters

2. we use a 6 parameter representation of rotations noting that rotation
matrices are just a simple unit vector-based orthogonal coordinate frame

3. the loss is based on both MSE of the 6-param representation as well as
a manifold-distance between matrix Representations

other training tricks could include a bigger range of initial rotations in 
the augmentation.
it could also be useful to incorporate some scale estmation ( for which i have
  some code ) but this introduces some weighting issues if it's all in the
  same network.  probably better to use a classic approach to get scale or a 
  different network.

we dont know, at this time, how well this rotation estimation network generalizes.
however, it will generalize if enough variability in training data is presented.


## automated landmarking

training code - `skull_mmd_vae_3d_no_test.R`

the inference code is shown in the file `auto_lm_mouse_ct_inference.R` 

we estimate landmark coordinates with unet-derived heatmaps (sigmoid activation) combined 
with direct regression of landmark positions.   this approach represents a bridge between 
heatmap coordinate regression methods and direct landmark regression.   we do not 
directly use the heatmaps in the loss function which greatly simplifies training.

it is possible that degenerate cases can occur when training from scratch.  the solution is to 
either train again from scratch with different weights, train with only slowly increasing 
amounts of augmentation or train with a linear activation first, then proceeding to relu, 
sigmoid or softmax only after good initial weights are found.  
or some combination of these approaches.

training this network, like the other one, uses augmentation.  however, this network 
assumes that the input image to be landmarked is roughly in the same orientation 
as the template.   again, this is not a theoretical constraint but a practical one in 
that training data, here, is fairly limited.  

potential experiments:

* weight individual landmarks differently in the loss functions to allow greater focus on the difficult landmarks.  this would amount to a fairly trivial change to the loss but not so trivial effort in evaluating how well it works or how to set the loss weights.   hyperparameter tuning would help but would also be fairly slow.  here is sample code ( which helped to some extent):

```
pointvec = tf$losses$MSE( datalist[[3]] * ptWeight, predpoints * ptWeight )
perptwt = rep( 1, nrow(ptList[[1]]) )
perptwt[ c(  1,  2, 29, 51, 52, 53 ) ] = 10.0
locptloss = tf$reduce_mean( pointvec * perptwt  )
```

* weight individual samples differently - for the same reasons as above.

*  pre-augment the data and save it in numpy arrays.  the current training wastes time on CPU.  
  augmenting first ( or in parallel to ) the training would allow the training loop to read directly from npy arrays thereby increasing training speed by 2 to possibly 10x.
  
* softmax activation - unsure if it would help or hurt but potentially worth investigating.

* training several landmarking networks with different reference templates - then perform an ensemble estimate of the final LM position.  clearly, the drawback is lots of extra training and some extra overhead at inference time.  one clear advantage of the current approach is speed ....

* image and landmark resolution - i chose 128,128,128 based primarily on computational considerations.   GPU RAM etc.  would we be better off learning something like 12 landmarks at 64x64x64 and then using these to initialize our solution at 128 cubed? murat noted that some "close together" landmarks are getting confused.  my initial experiments suggest that these worst-performing landmarks are consistently landmarks 1  2 29 51 52 53 (51, 52 seem to be worst) ... these could also be "focused on" by another layer of prediction.  so we could have ( if we went crazy ) a three level concatenated prediction framework that (i think) we could probably train end-to-end if we are are  a little bit clever about it.  it's ok if we cant, though, as it will still work nearly as well. level 0 would be 8 or 12 rough landmarks - which we could probably learn in rotationally invariant fashion (we've already shown we can do this with the rigid approach above).  level 1 would be what we have already.  level 2 would focus in on the problematic landmarks.

* post hoc image registration could also resolve this "landmark confusion" issue - probably pretty quickly in particular if we just run registration on the regions of interest, with proper initialization and constraints of course.

* priors for strain, species, etc would probably help the method generalize.  we worked out how to do this in the fish data and results are pretty nice.  in brief, we use an embedding layer to provide a generative source of priors and glue this onto the network as an input ( could be added in a variety of places ).  it also allows on the fly ensembling as one of the inputs to the embedding is a generative prior that allows one to naturally/smoothly perturb the solution space along controlled dimensions.   definitely interesting but potentially challenging to evaluate and adds some burden at inference time.  would need to justify by demonstrating real improvements with this approach. 

overall, more careful evaluation of the biological validity of the current results is warranted and will better guide what the next steps should be.
