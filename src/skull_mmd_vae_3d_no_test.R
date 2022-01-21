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
negateI <- function( x ) {
  max( x ) - x
}

read.fcsv<-function( x, skip=3 ) {
  df = read.table( x, skip=skip, sep=',' )
  colnames( df ) = c("id","x","y","z","ow","ox","oy","oz","vis","sel","lock","label","desc","associatedNodeID")
  return( df )
  }

prepro <- function( img, padder = 0 ) {
  # we  need to adjust the scale-space search for physical units, as below.
  img = (
    iMath(img, "Normalize") %>% iMath("PadImage",padder) ) * 255
  npt = c(1024,12) # ad-hoc choices here and in the next few lines - needs evaluation
  steppe = 4
  ss = scaleSpaceFeatureDetection( img, npt[1],
    stepsPerOctave=steppe,
    minScale = min( antsGetSpacing(img) )*0.1, # these select the physical space scale of features - smallest
    maxScale = min( antsGetSpacing(img) )*64,  # to largest
    negate = FALSE )$blobImage
  mask = thresholdImage( ss, 1e-9, Inf ) +
    randomMask( thresholdImage(img, 25, 255 ), npt[2] )

  print(paste("Extracted:",sum(mask==1),"mask points"))

  return( list( img =  img, ss = ss, mask = mask ) )
}

library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )

setwd("~/Desktop/MMLLM/")

set.seed( 2 )
fns = Sys.glob("data/*[0-9].nii.gz")
myids = basename( fns ) %>% tools::file_path_sans_ext(T)
demog = data.frame( ids = myids,
  fnimg=as.character(fns),
  fnseg=gsub( ".nii.gz","-LM.nii.gz", as.character(fns) ) )
demog$isTrain = TRUE
# now add in the new data
#fnsNew = Sys.glob("data_landmark_new_04_13_2021/*_.nii.gz")
#myidsNew = basename( fnsNew ) %>% tools::file_path_sans_ext(T)
#demogNew = data.frame( ids = myidsNew,
#  fnimg=as.character(fnsNew),
#  fnseg=paste0(as.character(fnsNew),".fcsv"))
#demogNew$isTrain = rep( TRUE, nrow(demogNew) )
#demog = rbind( demog, demogNew )

reoTemplate = antsImageRead( "templateImage.nii.gz" ) # antsImageClone( imgListTest[[templatenum]] )
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )# ptListTest[[templatenum]]
locdim = dim(reoTemplate)


numToTest = sum( demog$isTrain )
if ( ! exists( "imgList" ) ) {
  imgList = list()
  ptList = list()
  mynums = which( demog$isTrain )
  ct = 1
  for ( k in mynums ) {
    segisimg = length( grep("csv", demog$fnseg[k] ) ) == 0
    if ( segisimg ) {
      img = antsImageRead( demog$fnimg[k] )
      seg = antsImageRead( demog$fnseg[k] )
      pts = getCentroids( seg )[,1:img@dimension]
    } else {
      # the original model was trained on data with "bad" spacing
      # so we unfortunately propagate that here
      # not so bad b/c all we need is landmarks in the end which can be
      # expressed in whatever space the original image has.
      # the point is that we are throwing away image information in order to
      # express the points/images in voxel space - purposely to improve learning.
      img = antsImageRead( demog$fnimg[k] ) %>% resampleImage( locdim, useVoxels=TRUE)
      pts0 = data.matrix( read.fcsv( demog$fnseg[k] )[,2:4] )
      trad = sqrt( sum( antsGetSpacing( img )^2 ) )
      ptsi = makePointsImage( pts0, img*0+1, radius = trad ) %>% iMath("GD",1)
      invisible( antsCopyImageInfo2( img, reoTemplate ) )
      invisible( antsCopyImageInfo2( ptsi, reoTemplate ) )
      pts = getCentroids( ptsi )[,1:img@dimension]
    }
    if ( nrow( pts ) == 55 & min(rowSums(abs(pts))) > 0 ) {
      imgList[[ct]] = img
      ptList[[ct]] = pts
      cat(paste0(ct,"..."))
      ct = ct + 1
      }
    }
  }

nChannelsIn = 1
nChannelsOut = nrow( ptList[[1]] )
Sys.setenv(CUDA_VISIBLE_DEVICES=2)
mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
opre = paste0("models/autopointsupdate_sigmoid_128_weights_3d_checkpoints5_GPU",mygpu)
mdlfn = paste0(opre,'.h5')
mdlfnr = paste0(opre,'_recent.h5')
lossperfn = paste0(opre,'_training_history.csv')

# define an image template that lets us penalize rotations away from this reference
reoTemplate = antsImageRead( "templateImage.nii.gz" )
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )

###############################
generateData <- function( batch_size = 32, mySdAff=0.15, isTest=FALSE, verbose=FALSE ) {
  if ( isTest ) {
    batch_size = length( imgListTestReo )
    mySdAff = 0.01
    }
  nBlob = nChannelsOut
  X = array( dim = c( batch_size, locdim, nChannelsIn ) )
  CC = array( dim = c( batch_size, locdim, length(locdim) ) )
  Xp = array( dim = c( batch_size, nBlob, reoTemplate@dimension ) )
  ntxparams = reoTemplate@dimension*reoTemplate@dimension # rotation matrix
  Xtx = array( dim = c( batch_size, reoTemplate@dimension, reoTemplate@dimension ) )
  outdim = c( batch_size, locdim, nChannelsOut )
  if ( verbose ) {
    print(dim(X))
    print(dim(CC))
    print(dim(Xp))
  }
  imglist = list()
  for ( looper in 1:batch_size ) {
    whichSample = sample(1:length(imgList),1)
    if ( isTest ) {
      whichSample=looper
    }
    myseed = sample(1:100000000,1)
    if ( ! isTest ) locaff=mySdAff else locaff=0
    if ( ! isTest ) {
      temp = iMath(imgList[[whichSample]],"Normalize")
      blobs = ptList[[whichSample]]
      tx = fitTransformToPairedPoints( blobs, ptTemplate, "Rigid" )$transform
      temp = applyAntsrTransformToImage( tx, temp, reoTemplate  )
      blobs = applyAntsrTransformToPoint( invertAntsrTransform( tx ), blobs )
      } else {
      temp = iMath(imgListTestReo[[looper]],"Normalize")
      blobs = ptListTestReo[[looper]]
      }
    rr = randomAffineImage( temp, transformType='Rigid', sdAffine=locaff, seeder = myseed )
    loctx = rr[[2]]
    loctxi = invertAntsrTransform( loctx )
    blobs = applyAntsrTransformToPoint( loctxi, as.matrix( blobs ) )
    reftx=fitTransformToPairedPoints( blobs, ptTemplate, transformType='Rigid')$transform
    if ( verbose ) {
      print( dim(blobs))
      ptimg =  makePointsImage( blobs, rr[[1]], radius=2 ) %>% iMath("GD",2)
      if ( verbose ) plot( rr[[1]], ptimg )
    }
    coords = coordinateImages( rr[[1]] * 0 + 1 )
    Xp[looper,,]=blobs
    Xtx[looper,,] = matrix(getAntsrTransformParameters(reftx)[1:ntxparams],nrow=reoTemplate@dimension)
    X[looper,,,,1] = as.array( rr[[1]] )
    for ( jj in 1:reoTemplate@dimension) {
      CC[looper,,,,jj] = as.array( coords[[jj]] )
      }
    imglist[[looper]] = rr[[1]]
    }
  return( list(
    X,   # the images
    CC,  # coord conv
    Xp,  # the points
    Xtx, # the transforms
    imglist
  ) ) # input
}
########################
# Parameters --------------------------------------------------------------
K <- keras::backend()

# Model -------------------------------------------------------------------
unet = createUnetModel3D(
       list( NULL, NULL, NULL, nChannelsIn),
       numberOfOutputs = nChannelsOut,
       numberOfLayers = 4,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
       mode = c("regression")
     )
unet = keras_model( unet$inputs, tf$nn$sigmoid( unet$outputs[[1]] ) )
findpoints = deepLandmarkRegressionWithHeatmaps( unet, activation='relu', theta=NA )
if ( file.exists( mdlfn ) ) {
  print("Load from prior training history of this model")
  load_model_weights_hdf5( findpoints,  mdlfn )
} else if ( file.exists( "models/autopoints_relu_128_weights_3d_checkpoints5_GPU2_recent.h5" ) ) {
  print("Load from initial model")
  load_model_weights_hdf5( findpoints,  "models/autopoints_relu_128_weights_3d_checkpoints5_GPU2_recent.h5"  )
}


# testing data inference
# ggte = generateData( isTest=TRUE  )


testingError <- function( returnVector=FALSE, returnMatrix=FALSE, visualize=0 ) {
  with(tf$device("/cpu:0"), {
    telist = list(  tf$cast( ggte[[1]], mytype), tf$cast( ggte[[2]], mytype) )
    pointsoutte <- predict( findpoints, telist, batch_size = 4 )
    errvec = tf$losses$MSE( tf$cast( ggte[[3]], mytype), pointsoutte[[2]] )
  })
  if ( returnMatrix ) return( errvec )

    if ( visualize > 0 ) {
      for ( sss in sample( 1:nrow(ggte[[1]]), 1) ) {
        layout(matrix(1:2,nrow=2,byrow=T))
        intimg = as.antsImage( as.array( as.array(ggte[[1]]) )[sss,,,,1] ) %>%
              antsCopyImageInfo2( imgListTestReo[[sss]] )
        ptp = as.array(pointsoutte[[2]])[sss,,]
        ptimg = makePointsImage( ptp, imgListTestReo[[sss]] *0+1, radius=4 ) %>% iMath("GD",4)
        antsImageWrite( ptimg, '/tmp/z_pt_est.nii.gz')
        antsImageWrite( intimg, '/tmp/z_int_est.nii.gz')
        plot( intimg, ptimg, doCropping=F, axis=1,nslices=21,ncolumns=7)

        # real
        ptimg = makePointsImage( ptListTestReo[[sss]], imgListTestReo[[sss]]*0+1, radius=4 ) %>% iMath("GD",4)
        antsImageWrite( ptimg, '/tmp/z_pt_tru.nii.gz')
        plot( intimg, ptimg, doCropping=F, axis=1,nslices=21,ncolumns=7)
        errvecloc = sqrt( rowSums( (ptp-ptListTestReo[[sss]])^2 ) )
        lowct = sum( errvecloc < 25 )
        print(paste(visualize, mean(errvecloc), min(errvecloc), lowct ))
        }
      }

    if ( returnVector )
      return( colMeans( as.matrix( errvec ) ) )
    tferr = tf$reduce_mean( errvec )
    return( tf$cast( tferr, mytype ) )
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

compute_kernel <- function(x, y, sigma=tf$cast( 1e1, mytype ) ) {
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
num_epochs <- 5000
generator_batch_size = 1
local_batch_size = 1
lossper = data.frame( loss=NA, pt=NA, rot=NA, mmd=NA, mmdwt=NA, SD=NA, testErr=NA,
  ptrunningmean=NA )
epoch=1
if ( file.exists( lossperfn ) ) {
  lossper = read.csv( lossperfn )
  epoch = nrow( lossper ) + 1
  }
#optimizerE <- tf$keras$optimizers$Adam(1e-4)
optimizerE <- tf$keras$optimizers$Adam(1e-33) #for small voxel size.
mmdWeight = tf$cast( 10.0, mytype)
ptWeight = tf$cast( 4e-3, mytype )
rtWeight = tf$cast( 0.1, mytype )

# define reference points constant across batches
refpoints = tf$cast( ptTemplate, mytype )
refptlist = list()
for ( rfpt in 1:local_batch_size ) refptlist[[rfpt]] = refpoints
refpoints = tf$stack( refptlist )
##################################
layout(matrix(1:2,nrow=1,byrow=F))
##################################
for (epoch in epoch:num_epochs ) {
  locsd = 0.25 #  epoch / 2000  * 1.5a
  with(tf$device("/cpu:0"), {
    gg = generateData( batch_size = generator_batch_size,  mySdAff=locsd, verbose=FALSE  )
    datalist = list(
      tf$cast(gg[[1]],mytype), # images
      tf$cast(gg[[2]],mytype), # coord conv
      tf$cast(gg[[3]],mytype), # points
      tf$cast(gg[[4]],mytype) # transform parameters,
    )
  })

  total_loss = tf$cast(0.0,mytype)
  loss = tf$cast(0.0,mytype)
  pt_loss = tf$cast(0.0,mytype)
  loss_mmd = tf$cast(0.0,mytype)
  loss_rotation = tf$cast(0.0,mytype)
  with(tf$GradientTape(persistent = FALSE) %as% tape, {
      pointsout <- findpoints( datalist[1:2] )
      predpoints = tf$cast( pointsout[[2]], mytype )
      fittx = fitTransformToPairedPointsTF( predpoints, refpoints,
        numberOfPoints=nChannelsOut,
        reoTemplate@dimension,
        transformType = "Rigid",
        batch_size = local_batch_size,
        preventReflection = TRUE )
      for ( k in 1:local_batch_size) {
        loss_mmd = loss_mmd + compute_mmd( datalist[[3]][k,,], predpoints[k,,], takeMean=TRUE ) * mmdWeight
        }
      loss_mmd = tf$reduce_mean( loss_mmd )
      # use a general form for so3 distance as in: DOI 10.1007/s10851-009-0161-2
      # Section 3.5 https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
      idMat = tf$linalg$matmul(  datalist[[4]], datalist[[4]], transpose_b=TRUE )
      deltaMat = tf$linalg$matmul(  datalist[[4]], tf$cast( fittx, mytype ), transpose_b=TRUE )
      so3diff = idMat - deltaMat
      frobnormexp = tf$reduce_sum( tf$multiply(so3diff, so3diff), axis=list(1L,2L) )
      loss_rotation = loss_rotation + tf$reduce_mean( frobnormexp ) * rtWeight
      locptloss = tf$reduce_mean( tf$losses$MSE( datalist[[3]] * ptWeight, predpoints * ptWeight ) )
      pt_loss = pt_loss + locptloss
      loss = loss + loss_mmd  + pt_loss + loss_rotation
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
  lossper[epoch,'SD']=locsd
  loepoch = (epoch-100)
  if ( loepoch < 1 ) loepoch = 1
  lossper[epoch,'ptrunningmean']=mean(lossper[loepoch:(epoch-1),'pt'])
  if ( lossper[epoch,'ptrunningmean'] < min( lossper[loepoch:(epoch-1),'ptrunningmean'], na.rm=T  ) ) {
    print("best-recent")
    save_model_weights_hdf5( findpoints, mdlfnr )
    }
  write.csv( lossper, lossperfn, row.names=FALSE )
  if ( epoch %% 10 == 1 & epoch > 20 ) {
#    lossper[epoch,'testErr']=as.numeric( testingError( ) )
   if ( lossper[epoch,'ptrunningmean'] < min( lossper[10:(epoch-1),'ptrunningmean'], na.rm=T ) ) {
      print("best")
      save_model_weights_hdf5( findpoints, mdlfn )
      }
    }
  print( tail( lossper, 1 ) )
  }

