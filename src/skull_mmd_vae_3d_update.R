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
fns = Sys.glob("preprocessed/*skull.nii.gz")
myids = basename( fns )
myids = gsub( "skull.nii.gz", "", myids )
postfix = c( "reorient.nii.gz", "skulldist.nii.gz", "pts.csv" )
demog = data.frame(
  ids = myids,
  reo = paste0( myids, postfix[1]),
  dist = paste0( myids, postfix[2]),
  pts = paste0( myids, postfix[3]) )

demog = demog[ sample(1:nrow(demog),256),]
demog$isTrain = TRUE
demog$isTrain[ sample(1:nrow( demog ), 16 ) ] = FALSE

# define an image template that lets us penalize rotations away from this reference
reoTemplate = antsImageRead( "templateImage.nii.gz" )
newspc = antsGetSpacing( reoTemplate ) * 1.22
reoTemplate = resampleImage( reoTemplate, newspc ) %>%
  iMath("PadImage", 8 )%>%
  iMath("Normalize")
reoTemplate
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )
locdim = dim(reoTemplate)
lowerTrunc=1e-6
upperTrunc=0.98
numToTest = sum( demog$isTrain )
if ( ! exists( "imgList" ) ) {
  imgList = list()
  ptList = list()
  maskList = list()
  mynums = which( demog$isTrain )
  ct = 1
  for ( k in mynums ) {
    imgList[[ct]] = antsImageRead( paste0( "preprocessed/", demog$reo[k] ) )
    ptList[[ct]] = read.csv( paste0( "preprocessed/", demog$pts[k] ) )
    maskList[[ct]] = antsImageRead( paste0( "preprocessed/", demog$dist[k] ) )
    ct = ct + 1
    }
  }
##################################
if ( ! exists( "imgListTest" ) ) {
  mynums = which( !demog$isTrain )
  imgListTest = list()
  ptListTest = list()
  maskListTest = list()
  ct = 1
  for ( k in mynums ) {
    imgListTest[[ct]] = antsImageRead( paste0( "preprocessed/", demog$reo[k] ) )
    ptListTest[[ct]] = read.csv( paste0( "preprocessed/", demog$pts[k] ) )
    maskListTest[[ct]] = antsImageRead( paste0( "preprocessed/", demog$dist[k] ) )
    ct = ct + 1
    }
  }

gc()

mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
nChannelsIn = 1
nChannelsOut = nrow( ptListTest[[1]] )
opre = paste0("models/autopointsupdate_192_weights_3d_GPUsigmoidHRMask",mygpu)
mdlfn = paste0(opre,'.h5')
mdlfnr = paste0(opre,'_recent.h5')
lossperfn = paste0(opre,'_training_history.csv')


###############################
generateData <- function( batch_size = 32, mySdAff=0.15, isTest=FALSE, verbose=FALSE ) {
  if ( isTest ) {
    imgListLocal=imgListTest
    ptListLocal=ptListTest
    maskListLocal = maskListTest
  } else {
    imgListLocal=imgList
    ptListLocal=ptList
    maskListLocal = maskList
  }
  nBlob = nChannelsOut
  X = array( dim = c( batch_size, locdim, nChannelsIn ) )
  CC = array( dim = c( batch_size, locdim, length(locdim) ) )
  Xp = array( dim = c( batch_size, nBlob, reoTemplate@dimension ) )
  ntxparams = reoTemplate@dimension*reoTemplate@dimension # rotation matrix
  Xtx = array( dim = c( batch_size, reoTemplate@dimension, reoTemplate@dimension ) )
  Xmask = array( dim = c( batch_size, locdim, 1 ) )
  outdim = c( batch_size, locdim, nChannelsOut )
  if ( verbose ) {
    print(dim(X))
    print(dim(CC))
    print(dim(Xp))
  }
  gimglist = list()
  for ( looper in 1:batch_size ) {
    whichSample = sample(1:length(imgListLocal),1)
    myseed = sample(1:100000000,1)
    temp = iMath(imgListLocal[[whichSample]],"Normalize")
    blobs = data.matrix( ptListLocal[[whichSample]] )
    tx = fitTransformToPairedPoints( blobs, ptTemplate, "Rigid" )$transform
    temp = applyAntsrTransformToImage( tx, temp, reoTemplate  )
    blobs = applyAntsrTransformToPoint( invertAntsrTransform( tx ), blobs )
    rr = randomAffineImage( temp, transformType='ScaleShear', sdAffine=mySdAff, seeder = myseed )
    loctx = rr[[2]]
    loctxi = invertAntsrTransform( loctx )
    blobs = applyAntsrTransformToPoint( loctxi, as.matrix( blobs ) )
    reftx=fitTransformToPairedPoints( blobs, ptTemplate, transformType='Rigid')$transform
    coords = coordinateImages( rr[[1]] * 0 + 1 )
    Xp[looper,,]=blobs
    Xtx[looper,,] = matrix(getAntsrTransformParameters(reftx)[1:ntxparams],nrow=reoTemplate@dimension)
    X[looper,,,,1] = as.array( rr[[1]] )
    tempmask  = applyAntsrTransformToImage( tx, maskListLocal[[whichSample]], reoTemplate, interpolation='nearestNeighbor'  )
    Xmask[looper,,,,1] = as.array( tempmask )
    for ( jj in 1:reoTemplate@dimension) {
      CC[looper,,,,jj] = as.array( coords[[jj]] )
      }
    gimglist[[looper]] = rr[[1]]
    }
  return( list(
    X,   # the images
    CC,  # coord conv
    Xp,  # the points
    Xtx, # the transforms
    Xmask
  ) ) # input
}
########################
# make test data
ggte = generateData( batch_size=4,  mySdAff=0.1, isTest = TRUE )
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
# unetLM = keras_model( unetLM$inputs, layer_multiply( list( unetLM$output, unetLM$input ) ) )
findpoints = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none', theta=NA, useMask=TRUE )
if ( file.exists( mdlfn ) ) {
  print("Load from prior training history of this model")
  load_model_weights_hdf5( findpoints,  mdlfn )
}

testingError <- function(  ) {
  telist = list(  tf$cast( ggte[[1]], mytype), tf$cast( ggte[[2]], mytype), tf$cast( ggte[[5]], mytype) )
  pointsoutte <- predict( findpoints, telist, batch_size = 1 )
  errvec = tf$losses$MSE( tf$cast( ggte[[3]], mytype), pointsoutte[[2]] )
  errsum = as.numeric( tf$reduce_mean( errvec ) )
  # convert to image
  timg = as.antsImage( as.array( ggte[[1]][1,,,,1] ) ) %>% antsCopyImageInfo2( reoTemplate )
  mimg = as.antsImage( as.array( ggte[[5]][1,,,,1] ) ) %>% antsCopyImageInfo2( reoTemplate )
  ppts = pointsoutte[[2]][1,,]
  ptsi = makePointsImage( ppts, timg * 0 + 1, 0.5 )
  antsImageWrite( timg, '/tmp/temp.nii.gz' )
  antsImageWrite( mimg, '/tmp/tempm.nii.gz' )
  antsImageWrite( ptsi, '/tmp/tempp.nii.gz' )
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
generator_batch_size = 1
local_batch_size = 1
lossper = data.frame( loss=NA, pt=NA, rot=NA, mmd=NA, mmdwt=NA, SD=NA, testErr=NA,
  ptrunningmean=NA )
epoch=1
if ( file.exists( lossperfn ) ) {
  lossper = read.csv( lossperfn )
  epoch = nrow( lossper ) + 1
  }
optimizerEHi <- tf$keras$optimizers$Adam(1e-2)
optimizerEMid <- tf$keras$optimizers$Adam(5e-3)
optimizerELo <- tf$keras$optimizers$Adam(5e-4)
mmdWeight = tf$cast( 2e-3, mytype)
ptWeight = tf$cast( 1, mytype )
rtWeight = tf$cast( 0.2, mytype )

# define reference points constant across batches
refpoints = tf$cast( ptTemplate, mytype )
refptlist = list()
for ( rfpt in 1:local_batch_size ) refptlist[[rfpt]] = refpoints
refpoints = tf$stack( refptlist )
##################################
# layout(matrix(1:2,nrow=1,byrow=F))
##################################
for (epoch in epoch:num_epochs ) {
  locsd = 0.1
  ss = sample( 1:3 , 1 )
  if ( epoch %% 25 & ss == 1 ) # lower gradient step size when closer to good model
    optimizerE <- optimizerEHi
  if (  epoch %% 25 & ss == 2  ) # lower gradient step size when closer to good model
    optimizerE <- optimizerEMid
  if (  epoch %% 25 & ss == 3  ) # lower gradient step size when closer to good model
    optimizerE <- optimizerELo
  with(tf$device("/cpu:0"), {
    gg = generateData( batch_size = generator_batch_size,  mySdAff=locsd, verbose=FALSE  )
    datalist = list(
      tf$cast(gg[[1]],mytype), # images
      tf$cast(gg[[2]],mytype), # coord conv
      tf$cast(gg[[3]],mytype), # points
      tf$cast(gg[[4]],mytype), # transform parameters,
      tf$cast(gg[[5]],mytype) # mask
    )
  })


  total_loss = tf$cast(0.0,mytype)
  loss = tf$cast(0.0,mytype)
  pt_loss = tf$cast(0.0,mytype)
  loss_mmd = tf$cast(0.0,mytype)
  loss_rotation = tf$cast(0.0,mytype)
  loss_mask = tf$cast( 0.0, mytype )

  with(tf$GradientTape(persistent = FALSE) %as% tape, {
      pointsout <- findpoints( datalist[ c(1:2,5)] )
      predpoints = tf$cast( pointsout[[2]], mytype )
      fittx = fitTransformToPairedPointsTF( predpoints, refpoints,
        numberOfPoints=nChannelsOut,
        reoTemplate@dimension,
        transformType = "Rigid",
        batch_size = local_batch_size,
        preventReflection = TRUE )
      for ( k in 1:local_batch_size) {
#        if ( k == 1 ) { print( predpoints[k,50,] );  print( datalist[[3]][k,50,] ) }
        paireddist1 = compute_paired_distance( datalist[[3]][k,,], datalist[[3]][k,,] )
        paireddist2 = compute_paired_distance( predpoints[k,,], predpoints[k,,] )
        loss_mmd = loss_mmd + tf$reduce_mean( tf$keras$losses$MSE(paireddist1, paireddist2 ) ) * mmdWeight
        }
      # use a general form for so3 distance as in: DOI 10.1007/s10851-009-0161-2
      # Section 3.5 https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
      idMat = tf$linalg$matmul(  datalist[[4]], datalist[[4]], transpose_b=TRUE )
      deltaMat = tf$linalg$matmul(  datalist[[4]], tf$cast( fittx, mytype ), transpose_b=TRUE )
      so3diff = idMat - deltaMat
      frobnormexp = tf$reduce_sum( tf$multiply(so3diff, so3diff), axis=list(1L,2L) )
      loss_rotation = loss_rotation + tf$reduce_mean( frobnormexp ) * rtWeight
      locptloss = tf$reduce_mean( tf$losses$MSE( datalist[[3]] * ptWeight, predpoints * ptWeight ) )
      pt_loss = pt_loss + locptloss
      loss_mask = loss_mask + tf$reduce_mean(  pointsout[[1]] * datalist[[5]] ) * 1e5
      loss = loss + loss_mmd  + pt_loss + loss_rotation + loss_mask
    })
  unet_gradients <- tape$gradient(loss, findpoints$variables)
  optimizerE$apply_gradients(purrr::transpose(list(
      unet_gradients, findpoints$variables )))
  total_loss = total_loss + loss
  lossper[epoch,'loss']=as.numeric(total_loss)
  lossper[epoch,'pt']=as.numeric(pt_loss)
  lossper[epoch,'rot']=as.numeric(loss_rotation)
  lossper[epoch,'mmd']=as.numeric(loss_mmd)
  lossper[epoch,'mmdwt']=as.numeric(mmdWeight)
  lossper[epoch,'loss_mask']=as.numeric(loss_mask)
  lossper[epoch,'SD']=locsd
  loepoch = (epoch-100)
  if ( loepoch < 1 ) loepoch = 1
  lossper[epoch,'ptrunningmean']=mean(lossper[loepoch:(epoch-1),'pt'],na.rm=TRUE)
  if ( lossper[epoch,'ptrunningmean'] < min( lossper[loepoch:(epoch-1),'ptrunningmean'], na.rm=T  ) ) {
    print("best-recent")
    }
  write.csv( lossper, lossperfn, row.names=FALSE )
  if ( epoch %% 500 == 1 & epoch > 200 ) {
    with(tf$device("/cpu:0"), {
      lossper[epoch,'testErr']=testingError( )
      })
    if ( lossper[epoch,'testErr'] <= min( lossper[1:epoch,'testErr'], na.rm=TRUE ) ) {
      print("best")
#      save_model_weights_hdf5( findpoints, mdlfn )
      }
    }
  print( tail( lossper, 1 ) )
  }
