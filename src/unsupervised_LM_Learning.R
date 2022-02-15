Sys.setenv("TF_NUM_INTEROP_THREADS"=24)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=24)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=24)
library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( reticulate )
library( ggplot2 )
library( tfdatasets )
library( glue )
mytype = "float32"

# https://stackoverflow.com/questions/52992805/tensorflow-2d-histogram

randomMaskUlab <- function (img, nsamples, perLabel = FALSE, seed)
{
    img = check_ants(img)
    randmask = as.antsImage(array(0, dim = dim(img)), reference = img)
    if (perLabel == FALSE) {
        img[as.array(img) != 0] = 1
    }
    if (!missing(seed))
        set.seed(seed)
    ulabs <- sort(unique(c(as.numeric(img))))
    ulabs <- ulabs[ulabs > 0]
    for (ulab in ulabs) {
        ulabvec <- as.array(img) == as.numeric(ulab)
        n <- sum(ulabvec)
        k <- min(c(nsamples, n))
        ulabvec[-(sample(which(ulabvec), k))] = FALSE
        randmask[ulabvec] = 1:length(ulabvec)
    }
    return(randmask)
}


reparameterize <- function(mean, logvar) {
  eps <- k_random_normal(shape = mean$shape, dtype = tf$float32)
  eps * k_exp(logvar * 0.5) + mean
}


# Loss and optimizer ------------------------------------------------------

normal_loglik <- function(sample, mean, logvar ) {
  loglik <- k_constant(0.5) *
    (k_log(2 * k_constant(pi)) +
       logvar +
       k_exp(-logvar) * (sample - mean) ^ 2)
  - k_sum(loglik )
}


tf_cov <- function( x ) {
    mean_x = tf$reduce_mean(x, axis=0L, keepdims=TRUE)
    mx = tf$matmul(tf$transpose(mean_x), mean_x)
    vx = tf$matmul(tf$transpose(x), x)/tf$cast(tf$shape(x)[1], tf$float32)
    cov_xx = vx - mx
    return( cov_xx )
  }

compute_paired_distance <- function(x, y ) {
  x_size <- k_shape(x)[1]
  y_size <- k_shape(y)[1]
  dim <- k_shape(x)[2]
  tiled_x <- k_tile(
    k_reshape(x, k_stack(list(x_size, tf$cast(1,"int32"), dim))),
    k_stack(list(tf$cast(1,"int32"), y_size, tf$cast(1,"int32")))
  )
  tiled_y <- k_tile(
    k_reshape(y, k_stack(list(tf$cast(1,"int32"), y_size, dim))),
    k_stack(list(x_size, tf$cast(1,"int32"), tf$cast(1,"int32")))
  )
  k_mean( k_square(tiled_x - tiled_y), axis=3)
}


compute_kernel <- function(x, y, sigma=tf$cast( 0.01, mytype ) ) {
  x_size <- k_shape(x)[1]
  y_size <- k_shape(y)[1]
  dim <- k_shape(x)[2]
  tiled_x <- k_tile(
    k_reshape(x, k_stack(list(x_size, tf$cast(1,"int32"), dim))),
    k_stack(list(tf$cast(1,"int32"), y_size, tf$cast(1,"int32")))
  )
  tiled_y <- k_tile(
    k_reshape(y, k_stack(list(tf$cast(1,"int32"), y_size, dim))),
    k_stack(list(x_size, tf$cast(1,"int32"), tf$cast(1,"int32")))
  )
  sigmaterm = tf$cast( 2.0, mytype ) * k_square( sigma )
  k_exp(-k_mean(k_square(tiled_x - tiled_y)/sigmaterm, axis = 3) /
          k_cast(dim, tf$float32))
}

compute_mmd <- function( x, y, sigma=tf$cast( 1e1, mytype ), takeMean = FALSE ) {
  x_kernel <- compute_kernel(x, x, sigma=sigma )
  y_kernel <- compute_kernel(y, y, sigma=sigma )
  xy_kernel <- compute_kernel(x, y, sigma=sigma )
  if ( takeMean ) {
    myout = k_mean(x_kernel) + k_mean(y_kernel) - 2 * k_mean(xy_kernel)
  } else {
    myout = (x_kernel) + (y_kernel) - 2 * (xy_kernel)
  }
  return( myout )
}

binary_dice_loss = function (y_true, y_pred)
  {
  smoothing_factor = 1e-4
  K <- tensorflow::tf$keras$backend
  y_true_f = K$flatten(y_true)
  y_pred_f = K$flatten(y_pred)
  intersection = K$sum(y_true_f * y_pred_f)
  return(-1 * (2 * intersection + smoothing_factor)/(K$sum(y_true_f) +
        K$sum(y_pred_f) + smoothing_factor))
  }

makeHist <- function( x ) {
  h_true = tf$cast( tf$histogram_fixed_width( x, value_range=coordrange, nbins=16L), 'float32' )
  h_true / tf$reduce_sum( h_true )
}

makeHist3D <- function( x ) {
  coordrangex = range( ggte[[2]][,,,,1] )
  h_truex = tf$cast( tf$histogram_fixed_width( x[,1], value_range=coordrangex, nbins=16L), 'float32' )
  h_truex = h_truex / tf$reduce_sum( h_truex )
  coordrangex = range( ggte[[2]][,,,,2] )
  h_truey = tf$cast( tf$histogram_fixed_width( x[,2], value_range=coordrangex, nbins=16L), 'float32' )
  h_truey = h_truex / tf$reduce_sum( h_truey )
  coordrangex = range( ggte[[2]][,,,,3] )
  h_truez = tf$cast( tf$histogram_fixed_width( x[,3], value_range=coordrangex, nbins=16L), 'float32' )
  h_truez = h_truez / tf$reduce_sum( h_truez )
  tf$stack( list(h_truex, h_truey, h_truez ))
}

####################################################################################
################ Representation learning with MMD-VAE
################################################################
library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
set.seed( 2 )
prepdir = "/home/avants/data/DLBS/brainExtractionAndSegmentation/"
fns = Sys.glob( paste0( prepdir, "*brain.nii.gz") )[1:64]
myids = basename( fns )
myids = gsub( "_brain.nii.gz", "", myids )
demog = data.frame(
  ids = myids,
  fns= fns )
demog$isTrain = TRUE
numToTest = 8
demog$isTrain[ sample(1:nrow( demog ), numToTest ) ] = FALSE

# define an image template that lets us penalize rotations away from this reference
reoTemplate0 = antsImageRead( "~/data/brain_embeddings/T_template0.nii.gz" )
reoTemplate0 = cropImage( reoTemplate0 * thresholdImage( brainExtraction( reoTemplate0, 't1' ), 0.5, 1) )
newspc = antsGetSpacing( reoTemplate0 ) * 2
reoTemplate = resampleImage( reoTemplate0, newspc ) %>%
  iMath("PadImage",12) %>%
  iMath("Normalize") %>% padImageByFactor( 16 )
reoTemplate


ptsvd <- function( x ) {
  return( svd( x )$d )
}
# we use MMD to force the points to be distributed around the image
# the reference distribution is just randomly distributed
nChannelsIn = 1
nChannelsOut = 128L
txToLearn='Affine'
mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
if ( mygpu == 3 ) txToLearn='Rigid'
opre = paste0("models/learn_landmarks",mygpu,txToLearn)
mdlfn = paste0(opre,'.h5')
mdlfnr = paste0(opre,'_recent.h5')
lossperfn = paste0(opre,'_training_history.csv')

refptsR = ( randomMaskUlab( getMask( reoTemplate ), nChannelsOut, seed=00 ) %>%
    getCentroids() )[,1:3]
refptsimage = makePointsImage( refptsR, reoTemplate * 0 + 1 )

refpts = tf$cast( refptsR, 'float32' )
refptsc = refpts - tf$reduce_mean( refpts, axis=0L )
pp = ptsvd( refptsR )


locdim = dim(reoTemplate)
lowerTrunc=1e-6
upperTrunc=0.98
inferenceInds = c( 1:2 )
# if ( mygpu == 3 )
inferenceInds = c( 1:2, 4 )
numToTest = sum( !demog$isTrain )
if ( ! exists( "imgList" ) ) {
  imgList = list()
  mskList = list()
  mynums = which( demog$isTrain )
  ct = 1
  for ( k in mynums ) {
    img = antsImageRead( paste0(  demog$fns[k] ) )
    reg = antsRegistration( reoTemplate, img, "Affine"  )
    imgList[[ct]] = reg$warpedmovout
    mskList[[ct]] = getMask( reg$warpedmovout )
    ct = ct + 1
    }
  }
##################################
if ( ! exists( "imgListTest" ) ) {
  mynums = which( !demog$isTrain )
  imgListTest = list()
  mskListTest = list()
  ct = 1
  for ( k in mynums ) {
    img = antsImageRead( paste0(  demog$fns[k] ) )
    reg = antsRegistration( reoTemplate, img, "Affine" )
    imgListTest[[ct]] = ( reg$warpedmovout )
    mskListTest[[ct]] = getMask( reg$warpedmovout )
    ct = ct + 1
    }
  }

print("done collecting")
gc()



###############################
generateData <- function( batch_size = 32, mySdAff=0.15, isTest=FALSE,
    sameSample=FALSE, verbose=FALSE ) {
  set.seed( Sys.time() )
  if ( isTest ) {
    imgListLocal=imgListTest
    maskListLocal=mskListTest
  } else {
    imgListLocal=imgList
    maskListLocal=mskList
  }
  nBlob = nChannelsOut
  X = array( dim = c( batch_size, locdim, nChannelsIn ) )
  Xmask = array( dim = c( batch_size, locdim, 1 ) )
  CC = array( dim = c( batch_size, locdim, length(locdim) ) )
  ntxparams = reoTemplate@dimension*reoTemplate@dimension # rotation matrix
  Xtx = array( 0, dim = c( batch_size, reoTemplate@dimension, reoTemplate@dimension ) )
  gimglist = list()
  whichSamples = sample(1:length(imgListLocal),batch_size)
  if ( sameSample ) whichSamples = c(1,1)
  for ( looper in 1:batch_size ) {
    whichSample = whichSamples[ looper ]
    myseed = sample(1:100000000,1)
    if ( verbose ) print( paste( "whichSample", whichSample, "isTest", isTest, "seed",myseed ) )
    temp = iMath(imgListLocal[[whichSample]],"Normalize")
    if ( looper %% 2 == 0 ) {
      temp0 = iMath(imgListLocal[[whichSamples[ looper-1 ]]],"Normalize")
      rr = randomAffineImage( temp, transformType=txToLearn, sdAffine=mySdAff, seeder = myseed )
      loctx = rr[[2]]
      tempimg = applyAntsrTransformToImage( loctx, temp, reoTemplate  )
      reg = antsRegistration( temp0, tempimg, txToLearn )
      loctxi = readAntsrTransform( reg$fwdtransforms[1] )
      Xtx[looper,,] = matrix(getAntsrTransformParameters(loctxi)[1:ntxparams],nrow=reoTemplate@dimension)
      X[looper,,,,1] = as.array( tempimg )
      tempimg = applyAntsrTransformToImage( loctx, maskListLocal[[whichSample]], reoTemplate, interpolation = 'nearestNeighbor'  )
      Xmask[looper,,,,1] = as.array( tempimg )
      coords = coordinateImages( tempimg * 0 + 1 )
    } else {
      X[looper,,,,1] = as.array( temp )
      Xmask[looper,,,,1] = as.array(  maskListLocal[[whichSample]] )
      coords = coordinateImages( temp * 0 + 1 )
    }
    for ( jj in 1:reoTemplate@dimension) {
      CC[looper,,,,jj] = as.array( coords[[jj]] )
      }
    }
  gc()
  return( list(
    X,   # the images
    CC,  # coord conv
    Xtx, # the transforms,
    Xmask,
    loctxi
  ) ) # input
}
########################
# make test data
ggte = generateData( batch_size=8,  mySdAff=0.1, isTest = TRUE )
coordrange = range(ggte[[2]])
# Parameters --------------------------------------------------------------
K <- keras::backend()

# Model -------------------------------------------------------------------
unetLM = createUnetModel3D(
       list( NULL, NULL, NULL, nChannelsIn),
       numberOfOutputs = nChannelsOut,
       numberOfLayers = 5,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0.0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
#       mode = c("sigmoid")
       mode = c("classification")
     )
temp = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none',
  theta=NA, useMask=length(inferenceInds)==3  )
if ( length(inferenceInds)==3 ) {
  addout = keras::layer_add( tf$split( temp$outputs[[1]], 128L, axis=4L ) )
  findpoints = keras_model( temp$inputs,
      list( temp$outputs[[1]], temp$outputs[[2]], addout ) )
  }

if ( file.exists( mdlfn ) ) {
  print("Load from prior training history of this model")
  load_model_weights_hdf5( findpoints,  mdlfn )
} else {
  load_model_weights_hdf5( findpoints, "models/autopointsupdate_192_weights_3d_GPUsigmoidHR1.h5",skip_mismatch=T)
}

testingError <- function( verbose = FALSE ) {
  errsum=0.0
  for ( k in seq(1,1,by=2) ) {
  with(tf$device("/cpu:0"), {
      datalist1 = list(
        tf$cast( array( ggte[[1]][k,,,,], dim=c(1,locdim,1)) ,mytype), # images
        tf$cast( array( ggte[[2]][k,,,,], dim=c(1,locdim,3) ),mytype), # coord conv
        tf$cast( array( ggte[[3]][k,,], dim=c(1,3,3) ),mytype), # transform parameters,
        tf$cast( array( ggte[[4]][k,,,,], dim=c(1,locdim,1)) ,mytype) # images
      )
      datalist2 = list(
        tf$cast( array( ggte[[1]][k+1,,,,], dim=c(1,locdim,1)) ,mytype), # images
        tf$cast( array( ggte[[2]][k+1,,,,], dim=c(1,locdim,3) ),mytype), # coord conv
        tf$cast( array( ggte[[3]][k+1,,], dim=c(1,3,3) ),mytype), # transform parameters,
        tf$cast( array( ggte[[4]][k+1,,,,], dim=c(1,locdim,1)) ,mytype) # images
      )
      pointsout1 <- findpoints( datalist1[ inferenceInds ] )
      pointsout2 <- findpoints( datalist2[ inferenceInds ] )
      predpoints1 = tf$cast( pointsout1[[2]], mytype )
      predpoints2 = tf$cast( pointsout2[[2]], mytype )
      fittx = fitTransformToPairedPointsTF( predpoints1, predpoints2,
        numberOfPoints=tf$cast( nChannelsOut, "int32"),
        reoTemplate@dimension,
        transformType = txToLearn,
        batch_size = 1,
        preventReflection = TRUE )
      idMat = tf$linalg$matmul(  datalist2[[3]], datalist2[[3]], transpose_b=TRUE )
      deltaMat = tf$linalg$matmul(  datalist2[[3]], tf$cast( fittx, mytype ), transpose_b=TRUE )
      so3diff = idMat - deltaMat
      frobnormexp = tf$reduce_sum( tf$multiply(so3diff, so3diff), axis=list(1L,2L) )
      locloss = as.numeric( tf$reduce_mean( frobnormexp ) * rtWeight )
      if ( verbose ) print( paste( k, locloss ) )
      errsum = errsum + locloss
    })
  }
  return( errsum )
  }

refpthist = makeHist3D( refptsc )

# Training loop -----------------------------------------------------------
num_epochs <- 500000
generator_batch_size = 2
local_batch_size = 2
lossper = data.frame( loss=NA, rot=NA, mmd=NA, mmdwt=NA, SD=NA, testErr=NA )
epoch=1
if ( file.exists( lossperfn ) ) {
  lossper = read.csv( lossperfn )
  epoch = nrow( lossper ) + 1
  }
optimizerEHi <- tf$keras$optimizers$Adam(1e-3)
optimizerEMid <- tf$keras$optimizers$Adam(5e-4)
optimizerELo <- tf$keras$optimizers$Adam(1e-4)
optimizerE = optimizerEMid
mmdWeight = tf$cast( 1e-6, mytype)
ptWeight = tf$cast( 1, mytype )
rtWeight = tf$cast( 1.0, mytype )
htWeight = tf$cast( 0.01, mytype )
pdweight = tf$cast( 2e-2, mytype )
pdweightearly = tf$cast( 1e-4, mytype )
ss = 1
# refptsvd = tf$linalg$svd( refpts )[[1]]
# refptpairdist = tf$reduce_mean(  compute_paired_distance(refpts, refpts ) )
refptcov = tf_cov( refptsc )
for ( epoch in epoch:num_epochs ) {
  refptsR2 = ( randomMaskUlab( getMask( reoTemplate ), nChannelsOut, seed=999 ) %>%
      getCentroids() )[,1:3]
  refpts2 = tf$cast( refptsR2, 'float32' )
  refpts2c = refpts2 - tf$reduce_mean( refpts2, axis=0L )
  refpthist = makeHist( refpts2c )
  locsdmax = 0.2
  if ( txToLearn == 'Rigid') locsdmax=0.2
  locsd = locsdmax * sqrt( epoch ) / 500
  locsd = locsdmax
  if ( locsd > locsdmax ) locsd=locsdmax
  if ( epoch %% 25 == 0 ) ss = sample( 1:3 , 1 )
  if ( ss == 1 ) optimizerE <- optimizerEHi
  if ( ss == 2 ) optimizerE <- optimizerEMid
  if ( ss == 3 ) optimizerE <- optimizerELo
  if ( epoch < 100 ) mysame=TRUE else mysame=FALSE
  with(tf$device("/cpu:0"), {
    gg = generateData( batch_size = generator_batch_size,  mySdAff=locsd, verbose=TRUE, sameSample=mysame  )
    datalist1 = list(
      tf$cast( array( gg[[1]][1,,,,], dim=c(1,locdim,1)), mytype), # images
      tf$cast( array( gg[[2]][1,,,,], dim=c(1,locdim,3) ), mytype), # coord conv
      tf$cast( array( gg[[3]][1,,], dim=c(1,3,3) ), mytype), # transform parameters,
      tf$cast( array( gg[[4]][1,,,,], dim=c(1,locdim,1)) , mytype) # mask
    )
    datalist2 = list(
      tf$cast( array( gg[[1]][2,,,,], dim=c(1,locdim,1)) ,mytype), # images
      tf$cast( array( gg[[2]][2,,,,], dim=c(1,locdim,3) ),mytype), # coord conv
      tf$cast( array( gg[[3]][2,,], dim=c(1,3,3) ),mytype), # transform parameters,
      tf$cast( array( gg[[4]][2,,,,], dim=c(1,locdim,1)) ,mytype) # mask
    )
  })


  total_loss = tf$cast(0.0,mytype)
  loss = tf$cast(0.0,mytype)
  pt_loss = tf$cast(0.0,mytype)
  loss_mmd = tf$cast(0.0,mytype)
  loss_rotation = tf$cast(0.0,mytype)
  loss_mask = tf$cast( 0.0, mytype )
  refptstx = tf$cast( applyAntsrTransformToPoint( gg[[5]], refptsR ), 'float32')

  with(tf$GradientTape(persistent = FALSE) %as% tape, {
      pointsout1 <- findpoints( datalist1[ inferenceInds ] )
      pointsout2 <- findpoints( datalist2[ inferenceInds ] )
      predpoints1 = tf$cast( pointsout1[[2]], mytype )
      predpoints2 = tf$cast( pointsout2[[2]], mytype )
      # predpoints1=tf$reshape( refpts,  c(1L,128L,3L) )  # for verification
      # predpoints2=tf$reshape( refptstx, c(1L,128L,3L) )  # for verification
      predpoints1 = predpoints1 - tf$reduce_mean( predpoints1, axis=1L, keepdims=TRUE )
      predpoints2 = predpoints2 - tf$reduce_mean( predpoints2, axis=1L, keepdims=TRUE  )
      fittx = fitTransformToPairedPointsTF( predpoints1, predpoints2,
        numberOfPoints=tf$cast( nChannelsOut, "int32"),
        reoTemplate@dimension,
        transformType = txToLearn,
        batch_size = 1,
        preventReflection = TRUE )
      fwdmat = tf$cast(fittx[1,,],'float32')
      invmat = tf$linalg$inv( fwdmat )
      predpoints2 = tf$linalg$matmul( tf$cast(predpoints2[1,,],'float'), fwdmat, transpose_b=TRUE )
      for ( k in 1:1) {
#        loss1 = tf$losses$MSE( paireddistref, paireddist1 ) %>% tf$reduce_mean()
#        loss2 = tf$losses$MSE( paireddistref, paireddist2 ) %>% tf$reduce_mean()
#        paireddist3 = compute_mmd( predpoints2[k,,], predpoints1[k,,] )%>% tf$reduce_mean()
#        paireddist1 = compute_mmd( refpts, predpoints1[k,,] )%>% tf$reduce_mean()
#        paireddist2 = compute_mmd( refpts, predpoints2[k,,] )%>% tf$reduce_mean()
#        ptsvd1 = tf$linalg$svd(  predpoints1[k,,] )[[1]]
#        ptsvd2 = tf$linalg$svd(  predpoints2[k,,] )[[1]]
#        loss_mmd = tf$reduce_mean(
#                    tf$losses$MSE(refptsvd,ptsvd1) +
#                    tf$losses$MSE(refptsvd,ptsvd2)  ) * mmdWeight
        loss_mmd = tf$reduce_mean(
            tf$losses$MSE(refptcov, tf_cov( predpoints1[1,,] ) ) +
            tf$losses$MSE(refptcov, tf_cov( predpoints2[,] ) )  ) * mmdWeight
        pt_loss_early = (
          tf$losses$MSE( refptsc, predpoints1 ) %>% tf$reduce_mean( )  +
          tf$losses$MSE( refptsc, predpoints2 ) %>% tf$reduce_mean( )
        ) * pdweightearly
        pt_loss = tf$reduce_mean(
          tf$keras$losses$kullback_leibler_divergence( refpthist, makeHist3D( predpoints1[1,,] ) )  +
          tf$keras$losses$kullback_leibler_divergence( refpthist, makeHist3D( predpoints2 ) )  )  * pdweight
        }
      # use a general form for so3 distance as in: DOI 10.1007/s10851-009-0161-2
      # Section 3.5 https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
      idMat = tf$linalg$matmul(  datalist2[[3]], datalist2[[3]], transpose_b=TRUE )
      deltaMat = tf$linalg$matmul(  datalist2[[3]], tf$cast( fittx, mytype ), transpose_b=TRUE )
      so3diff = idMat - deltaMat
      frobnormexp = tf$reduce_sum( tf$multiply(so3diff, so3diff), axis=list(1L,2L) )
      loss_rotation = loss_rotation + tf$reduce_mean( frobnormexp ) * rtWeight
      loss_mask = 0.0 # ( binary_dice_loss( datalist1[[4]], pointsout1[[3]]) +
      #  binary_dice_loss( datalist2[[4]], pointsout2[3] )  ) * htWeight
      loss = loss_rotation  + pt_loss + pt_loss_early / epoch^0.2
    })

  unet_gradients <- tape$gradient(loss, findpoints$variables)
  optimizerE$apply_gradients(purrr::transpose(list(
      unet_gradients, findpoints$variables )))
  total_loss = total_loss + loss
  lossper[epoch,'loss']=as.numeric(total_loss)
  lossper[epoch,'rot']=as.numeric(loss_rotation)
  lossper[epoch,'mmd']=as.numeric(loss_mmd)
  lossper[epoch,'mmdwt']=as.numeric(mmdWeight)
  lossper[epoch,'loss_mask']=as.numeric(loss_mask)
  lossper[epoch,'loss_pdist']=as.numeric(pt_loss)
  lossper[epoch,'early']=as.numeric(pt_loss_early)
  lossper[epoch,'SD']=locsd
  lossper[epoch,'whichopt']=ss
  loepoch = (epoch-100)

  write.csv( lossper, lossperfn, row.names=FALSE )

  if ( epoch %% 200 == 1 & epoch > 20 ) {
    lossper[epoch,'testErr']=testingError( )
    if ( lossper[epoch,'testErr'] <= min( lossper[1:epoch,'testErr'], na.rm=TRUE ) ) {
      print("best")
      save_model_weights_hdf5( findpoints, mdlfn )
      }
    }
  if ( epoch %% 20 == 1 & epoch > 20 ) {
    save_model_weights_hdf5( findpoints, gsub("marks","marks_recent",mdlfn) )
    }

  print( tail( lossper[1:epoch,], 1 ) )
  }

mdlfn="models/learn_landmarks2.h5"
mdlfn="models/learn_landmarks3.h5"
mdlfn="models/learn_landmarks_recent2.h5"

unetLM = createUnetModel3D(
       list( NULL, NULL, NULL, nChannelsIn),
       numberOfOutputs = nChannelsOut,
       numberOfLayers = 5,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0.0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
#       mode = c("sigmoid")
       mode = c("classification")
     )
temp = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none',
  theta=NA, useMask=length(inferenceInds)==3  )
if ( length(inferenceInds)==3 ) {
  addout = keras::layer_add( tf$split( temp$outputs[[1]], 128L, axis=4L ) )
  findpoints = keras_model( temp$inputs,
      list( temp$outputs[[1]], temp$outputs[[2]], addout ) )
  }


mdlfn="models/learn_landmarks_recent3.h5"
mdlfn='models/learn_landmarks_recent3.h5'
for ( k in c(1,3,5,7) ) {
  mdlfn='models/learn_landmarks_recent2Affine.h5'
  load_model_weights_hdf5( findpoints,  mdlfn )
  print(k)
  im1 = as.antsImage( ggte[[1]][k,,,,1] ) %>% antsCopyImageInfo2( reoTemplate )
  im2 = as.antsImage( ggte[[1]][k+1,,,,1] ) %>% antsCopyImageInfo2( reoTemplate )
  antsImageWrite( im1, '/tmp/temp0.nii.gz')
  antsImageWrite( im2, '/tmp/temp1.nii.gz')
  datalist1 = list(
    tf$cast( array( ggte[[1]][k,,,,], dim=c(1,locdim,1)) ,mytype), # images
    tf$cast( array( ggte[[2]][k,,,,], dim=c(1,locdim,3) ),mytype), # coord conv
    tf$cast( array( ggte[[3]][k,,], dim=c(1,3,3) ),mytype), # transform parameters,
    tf$cast( array( ggte[[4]][k,,,,], dim=c(1,locdim,1)) ,mytype) # images
  )
  datalist2 = list(
    tf$cast( array( ggte[[1]][k+1,,,,], dim=c(1,locdim,1)) ,mytype), # images
    tf$cast( array( ggte[[2]][k+1,,,,], dim=c(1,locdim,3) ),mytype), # coord conv
    tf$cast( array( ggte[[3]][k+1,,], dim=c(1,3,3) ),mytype), # transform parameters,
    tf$cast( array( ggte[[4]][k+1,,,,], dim=c(1,locdim,1)) ,mytype) # images
  )
  pointsout1 <- findpoints( datalist1[ inferenceInds ] )
  pointsout2 <- findpoints( datalist2[ inferenceInds ] )
  predpoints1 = tf$cast( pointsout1[[2]], mytype )
  predpoints2 = tf$cast( pointsout2[[2]], mytype )
  rrr <- RANSACAlt(
    as.array(predpoints1)[1,,],
    as.array(predpoints2)[1,,], transformType = 'Rigid',
    nToTrim = 1, minProportionPoints = 0.11, nCVGroups=4 )
  im2map = applyAntsrTransformToImage( rrr$finalModel$transform, im2, im1 )
  print( paste("MI0:", antsImageMutualInformation(im1,im2),"MIMRansac",antsImageMutualInformation(im1,im2map)))
  tx = fitTransformToPairedPoints( as.array(predpoints2)[1,,], as.array(predpoints1)[1,,], "Rigid")$transform
  im2map = applyAntsrTransformToImage(tx, im2, im1 )
  print( paste("MI0:", antsImageMutualInformation(im1,im2),"MIMFull",antsImageMutualInformation(im1,im2map)))
  # antsImageWrite( im2map, '/tmp/temp1w.nii.gz')
  if ( FALSE ) {
    pts0 = makePointsImage( as.array(predpoints1)[1,,], im1 * 0 + 1 )
    pts1 = makePointsImage( as.array(predpoints2)[1,,], im2 * 0 + 1 )
    antsImageWrite( pts0, '/tmp/temp0p.nii.gz')
    antsImageWrite( pts1, '/tmp/temp1p.nii.gz')
  }
  fittx = fitTransformToPairedPointsTF( predpoints1, predpoints2,
    numberOfPoints=tf$cast( nChannelsOut, "int32"),
    reoTemplate@dimension,
    transformType = txToLearn,
    batch_size = 1,
    preventReflection = TRUE )
  predpoints1c = predpoints1 - tf$reduce_mean( predpoints1, axis=1L, keepdims=TRUE )
  predpoints2c = predpoints2 - tf$reduce_mean( predpoints2, axis=1L, keepdims=TRUE  )
  refptsc = refpts - tf$reduce_mean( refpts, axis=0L )
  print(tf$reduce_mean( tf$math$abs( refptsc - predpoints1c )))
  print(tf$reduce_mean( tf$math$abs( refptsc - predpoints2c )))
  fwdmat = tf$cast(fittx[1,,],'float32')
  invmat = tf$linalg$inv( fwdmat )
  pptx =  tf$linalg$matmul( tf$cast(predpoints2c[1,,],'float'), fwdmat, transpose_b=TRUE )
  print(tf$reduce_mean( tf$math$abs( refptsc - pptx )))
print(  ggte[[3]][1,,])
print(  fittx )
}


testingError(verbose=T)


# testing mapping questions
ch2=antsImageRead( getANTsRData( "ch2" ) )
msk=getMask(ch2)
reoTemplate=ch2
rmsk=randomMaskUlab( msk, 128 )
rpts = getCentroids( rmsk )[,1:3]
for ( txToLearn in c("Rigid" , "Affine") ) {
  rr = randomAffineImage( ch2, transformType=txToLearn, sdAffine=0.1, seeder = 1L )
  loctx = rr[[2]]
  tempimg = applyAntsrTransformToImage( loctx, ch2, reoTemplate  )
  reg = antsRegistration( ch2, tempimg, txToLearn )
  loctxireg = readAntsrTransform( reg$fwdtransforms[1] )
  loctxi = invertAntsrTransform( loctx )

  rptsrot = applyAntsrTransformToPoint( loctxi, rpts )
  # mpi = makePointsImage( rptsrot, getMask(rr[[1]]) )
  # rpts = fixed
  predpoints2=tf$reshape( tf$cast( rpts, 'float32'), c(1L,128L,3L) )
  # rptsrot = moving
  predpoints1=tf$reshape( tf$cast( rptsrot, 'float32'), c(1L,128L,3L) )
  fittx = fitTransformToPairedPointsTF( predpoints1, predpoints2,
    numberOfPoints=tf$cast( 128, "int32"),
    3,
    transformType = txToLearn,
    batch_size = 1,
    preventReflection = TRUE )
  fwdmat = tf$cast(fittx[1,,],'float32')
  invmat = tf$linalg$inv( fwdmat )
  predpoints1c = predpoints1 - tf$reduce_mean( predpoints1, axis=1L, keepdims=TRUE )
  predpoints2c = predpoints2 - tf$reduce_mean( predpoints2, axis=1L, keepdims=TRUE  )
  refpts=predpoints1
  refptsc = refpts - tf$reduce_mean( refpts, axis=1L, keepdims=TRUE  )
  print(tf$reduce_mean( tf$math$abs( refptsc - predpoints1c )))
  print(tf$reduce_mean( tf$math$abs( refptsc - predpoints2c )))
  if ( txToLearn == 'Rigid' )
    pptx =  tf$linalg$matmul( tf$cast(predpoints2c[1,,],'float'), fwdmat, transpose_b=TRUE )
  if ( txToLearn == 'Affine' )
    pptx =  tf$linalg$matmul( tf$cast(predpoints2c[1,,],'float'), fwdmat, transpose_b=TRUE )
  print(paste("should be zero-ish",txToLearn))
  print(tf$reduce_mean( tf$math$abs( refptsc - pptx )))
}
