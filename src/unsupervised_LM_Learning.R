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
reoTemplate = antsImageRead( "~/data/brain_embeddings/T_template0.nii.gz" )
reoTemplate = reoTemplate * thresholdImage( brainExtraction( reoTemplate, 't1' ), 0.5, 1)
newspc = antsGetSpacing( reoTemplate ) * 3
reoTemplate = resampleImage( reoTemplate, newspc ) %>%
  iMath("PadImage", 16 )%>%
  iMath("Normalize") %>% padImageByFactor( 16 )
reoTemplate
locdim = dim(reoTemplate)
lowerTrunc=1e-6
upperTrunc=0.98
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

mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
nChannelsIn = 1
nChannelsOut = 55
opre = paste0("models/learn_landmarks",mygpu)
mdlfn = paste0(opre,'.h5')
mdlfnr = paste0(opre,'_recent.h5')
lossperfn = paste0(opre,'_training_history.csv')


###############################
generateData <- function( batch_size = 32, mySdAff=0.15, isTest=FALSE, verbose=FALSE ) {
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
  for ( looper in 1:batch_size ) {
    whichSample = sample(1:length(imgListLocal),1)
    if ( verbose ) print( paste( "whichSample", whichSample, "isTest", isTest ) )
    myseed = sample(1:100000000,1)
    temp = iMath(imgListLocal[[whichSample]],"Normalize")
    if ( looper %% 2 == 1 ) {
      rr = randomAffineImage( temp, transformType=sample( c('ScaleShear','Rigid'))[1], sdAffine=mySdAff, seeder = myseed )
      loctx = rr[[2]]
      loctxi = invertAntsrTransform( loctx )
      Xtx[looper,,] = matrix(getAntsrTransformParameters(loctxi)[1:ntxparams],nrow=reoTemplate@dimension)
      tempimg = applyAntsrTransformToImage( loctx, temp, reoTemplate  )
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
    Xmask
  ) ) # input
}
########################
# make test data
ggte = generateData( batch_size=8,  mySdAff=0.1, isTest = TRUE )
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
       mode = c("sigmoid")
     )
findpoints = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none', theta=NA, useMask=TRUE )

if ( file.exists( mdlfn ) ) {
  print("Load from prior training history of this model")
  load_model_weights_hdf5( findpoints,  mdlfn )
}

testingError <- function( verbose = FALSE ) {
  errsum=0.0
  with(tf$device("/cpu:0"), {
    for ( k in seq(1,8,by=2) ) {
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
        transformType = "Affine",
        batch_size = 1,
        preventReflection = TRUE )
      idMat = tf$linalg$matmul(  datalist1[[3]], datalist1[[3]], transpose_b=TRUE )
      deltaMat = tf$linalg$matmul(  datalist1[[3]], tf$cast( fittx, mytype ), transpose_b=TRUE )
      so3diff = idMat - deltaMat
      frobnormexp = tf$reduce_sum( tf$multiply(so3diff, so3diff), axis=list(1L,2L) )
      locloss = as.numeric( tf$reduce_mean( frobnormexp ) * rtWeight )
      if ( verbose ) print( paste( k, locloss ) )
      errsum = errsum + locloss
      }
    })
  return( errsum )
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
optimizerEHi <- tf$keras$optimizers$Adam(1e-4)
optimizerEMid <- tf$keras$optimizers$Adam(5e-3)
optimizerELo <- tf$keras$optimizers$Adam(5e-4)
optimizerE = optimizerEMid
mmdWeight = tf$cast( 2e-1, mytype)
ptWeight = tf$cast( 1, mytype )
rtWeight = tf$cast( 0.2, mytype )
htWeight = tf$cast( 5e0, mytype )

# we use MMD to force the points to be distributed around the image
# the reference distribution is just randomly distributed
refpts = randomMask( getMask( reoTemplate ), nChannelsOut, seed=11 ) %>% labelClusters(1) %>% getCentroids()
stopifnot( nrow(refpts)== nChannelsOut)
refpts = tf$cast( refpts[,1:3], 'float32' )
ss = 1

for ( epoch in epoch:num_epochs ) {
  locsd = 0.25
  if ( epoch %% 205 == 0 ) ss = sample( 1:3 , 1 )
  if ( ss == 1 ) optimizerE <- optimizerEHi
  if ( ss == 2 ) optimizerE <- optimizerEMid
  if ( ss == 3 ) optimizerE <- optimizerELo
  with(tf$device("/cpu:0"), {
    gg = generateData( batch_size = generator_batch_size,  mySdAff=locsd, verbose=TRUE  )
    datalist1 = list(
      tf$cast( array( gg[[1]][1,,,,], dim=c(1,locdim,1)) ,mytype), # images
      tf$cast( array( gg[[2]][1,,,,], dim=c(1,locdim,3) ),mytype), # coord conv
      tf$cast( array( gg[[3]][1,,], dim=c(1,3,3) ),mytype), # transform parameters,
      tf$cast( array( gg[[4]][1,,,,], dim=c(1,locdim,1)) ,mytype) # mask
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

  with(tf$GradientTape(persistent = FALSE) %as% tape, {
      pointsout1 <- findpoints( datalist1[ inferenceInds ] )
      pointsout2 <- findpoints( datalist2[ inferenceInds ] )
      predpoints1 = tf$cast( pointsout1[[2]], mytype )
      predpoints2 = tf$cast( pointsout2[[2]], mytype )
      fittx = fitTransformToPairedPointsTF( predpoints1, predpoints2,
        numberOfPoints=tf$cast( nChannelsOut, "int32"),
        reoTemplate@dimension,
        transformType = "Affine",
        batch_size = 1,
        preventReflection = TRUE )
      for ( k in 1:1) {
#        if ( k == 1 ) { print( predpoints[k,50,] );  print( datalist[[3]][k,50,] ) }
        paireddist1 = compute_mmd( refpts, predpoints2[k,,] )
        paireddist2 = compute_mmd( refpts, predpoints1[k,,] )
        loss_mmd = (tf$reduce_mean( paireddist1 ) + tf$reduce_mean( paireddist2 )  ) * mmdWeight
        }
      # use a general form for so3 distance as in: DOI 10.1007/s10851-009-0161-2
      # Section 3.5 https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
      idMat = tf$linalg$matmul(  datalist1[[3]], datalist1[[3]], transpose_b=TRUE )
      deltaMat = tf$linalg$matmul(  datalist1[[3]], tf$cast( fittx, mytype ), transpose_b=TRUE )
      so3diff = idMat - deltaMat
      frobnormexp = tf$reduce_sum( tf$multiply(so3diff, so3diff), axis=list(1L,2L) )
      loss_rotation = loss_rotation + tf$reduce_mean( frobnormexp ) * rtWeight
      loss = loss_mmd  + loss_rotation # + loss_mask
    })

  unet_gradients <- tape$gradient(loss, findpoints$variables)
  optimizerE$apply_gradients(purrr::transpose(list(
      unet_gradients, findpoints$variables )))
  total_loss = total_loss + loss
  lossper[epoch,'loss']=as.numeric(total_loss)
  lossper[epoch,'rot']=as.numeric(loss_rotation)
  lossper[epoch,'mmd']=as.numeric(loss_mmd)
  lossper[epoch,'mmdwt']=as.numeric(mmdWeight)
  lossper[epoch,'SD']=locsd
  lossper[epoch,'whichopt']=ss
  loepoch = (epoch-100)

  write.csv( lossper, lossperfn, row.names=FALSE )

  if ( epoch %% 10 == 1 & epoch > 20 ) {
    lossper[epoch,'testErr']=testingError( )
    if ( lossper[epoch,'testErr'] <= min( lossper[1:epoch,'testErr'], na.rm=TRUE ) ) {
      print("best")
      save_model_weights_hdf5( findpoints, mdlfn )
      }
    }

  print( tail( lossper, 1 ) )
  }

x=c(14,27)
im1 = as.antsImage( gg[[1]][1,,,,1] ) %>% antsCopyImageInfo2( reoTemplate )
im2 = as.antsImage( gg[[1]][2,,,,1] ) %>% antsCopyImageInfo2( reoTemplate )
antsImageWrite( im1, '/tmp/temp0.nii.gz')
antsImageWrite( im2, '/tmp/temp1.nii.gz')
pts0 = makePointsImage( as.array(predpoints1)[1,,], imgList[[1]] * 0 + 1 )
pts1 = makePointsImage( as.array(predpoints2)[1,,], imgList[[1]] * 0 + 1 )
antsImageWrite( pts0, '/tmp/temp0p.nii.gz')
antsImageWrite( pts1, '/tmp/temp1p.nii.gz')
