
Sys.setenv("TF_NUM_INTEROP_THREADS"=12)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=12)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=12)
library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( reticulate )
library( ggplot2 )
library( tfdatasets )
library( glue )
library( pracma ) # just for cross function ie vector cross product
mytype = "float32"

####################################################################################
################ SEE: Representation learning with MMD-VAE
# SEE:  An Analysis of SVD for Deep Rotation Estimation
# "Interestingly, SVD- Inference slightly outperforms SVD-Train, which suggests
# in this scenario where viewpoints have a non-uniform prior, training a network
# to regress directly to the desired rotation can work well."
# also:
# The results of the previous sections are broad and conclusive: a continuous 9D
# unconstrained rep- resentation followed by an SVD projection onto SO(3) is
# consistently an effective, and often the state-of-the-art, representation for
# 3D rotations in neural networks. It is usable in a variety of ap- plication
# settings including without supervision. The strong empirical evidence is
# supported by a theoretical analysis that supports SVD as the preferred
# projection onto SO(3).
# SEE: On the Continuity of Rotation Representations in Neural Networks
# SEE: for how to do a layer
# https://stackoverflow.com/questions/61178106/how-to-use-svd-inside-keras-layers
################################################################
negateI <- function( x ) {
  max( x ) - x
}


set.seed( 2 )
fns = Sys.glob("preprocessed4/*rec_reorient.nii.gz")
myids = basename( fns ) %>% tools::file_path_sans_ext(T)
demog = data.frame( ids = myids,
  fnimg=as.character(fns)  )
demog$isTrain = TRUE
demog$isTrain[ sample(1:nrow( demog ),16) ] = FALSE

# define an image template that lets us penalize rotations away from this reference
# templatenum=10
reoTemplate = antsImageRead( "templateImage.nii.gz" ) %>% iMath("Normalize")
newspc = 2.76 * antsGetSpacing( reoTemplate )
reoTemplate = resampleImage( reoTemplate, newspc ) %>% iMath( "PadImage", 32 )
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )# ptListTest[[templatenum]]
locdim = dim(reoTemplate)


numToTest = sum( demog$isTrain )
if ( ! exists( "imgList" ) ) {
  imgList = list()
  mynums = which( demog$isTrain )
  ct = 1
  for ( k in mynums ) {
    img = antsImageRead( demog$fnimg[k] )
    imgList[[ct]] = img
    cat(paste0(ct,"..."))
    ct = ct + 1
    }
  }

if ( ! exists( "imgListTest" ) ) {
  imgListTest = list()
  mynums = which( ! demog$isTrain )
  ct = 1
  for ( k in mynums ) {
    img = antsImageRead( demog$fnimg[k] )
    imgListTest[[ct]] = img
    cat(paste0(ct,"..."))
    ct = ct + 1
    }
  }

###############################
generateData <- function( batch_size = 32, mySdAff=0.15, isTest=FALSE, verbose=FALSE ) {
  if ( isTest ) {
    imgListLocal = imgListTest
  } else {
    imgListLocal = imgList
  }
  locdim = dim( reoTemplate )
  X = array( dim = c( batch_size, locdim, 1 ) )
  Y = array( dim = c( batch_size, 6 ) )
  matlist=list()
  for ( looper in 1:batch_size ) {
    whichSample = sample(1:length(imgListLocal),1)
    myseed = sample(1:100000000,1)
    locaff=mySdAff
    temp = iMath( imgListLocal[[whichSample]], "Normalize" )
    rr = randomAffineImage( temp, "Rigid", sdAffine=locaff, seeder = myseed )
    loctx = rr[[2]]
    loctxi = invertAntsrTransform( loctx )
    Y[looper,] = getAntsrTransformParameters(loctxi)[1:6]
    transformed = applyAntsrTransformToImage( loctx, temp, reoTemplate  )
    X[looper,,,,1] = as.array( transformed )
    rotmat = matrix( getAntsrTransformParameters( loctxi )[1:9], nrow = 3 )
    matlist[[looper]] = rotmat
    }
  return( list(
    X,   # the images
    Y, matlist
  ) ) # input
}

########################
# Parameters --------------------------------------------------------------
K <- keras::backend()
ggte = generateData( batch_size=8, mySdAff = 5, isTest=TRUE  )

tfRotVectorToMatrix <- function( x ) {
  # generate predicted rotation matrices
  bsz = x$shape[[1]]
  rmatstrunc = tf$linalg$matrix_transpose( k_reshape( x, c(bsz, 2, 3 ) ) )
  mylist = list()
  for ( k in 1:bsz ) {
    vec3 = tf$linalg$cross( rmatstrunc[k,,1], rmatstrunc[k,,2] )
    temp = tf$concat( list(rmatstrunc[k,,], k_reshape(vec3,c(3,1))) , axis=1L)
    tt = tf$linalg$svd(temp)
    d = as.numeric( tf$linalg$det(tf$matmul(tt[[3]], tt[[2]], transpose_b=F ) ) )
    mydiag = tf$cast( tf$linalg$diag( c( 1.0, 1.0, d ) ), 'float32' )
    newu = tf$cast( tf$matmul( tt[[2]], mydiag ), "float32" )
    temp = tf$linalg$matmul( newu, tt[[3]], transpose_b=TRUE)
    mylist[[k]] = temp
  }
  tf$stack( mylist )
}

SO3_6_param_loss <- function( x, xpred ) {
  xR = tfRotVectorToMatrix( x )
  xpredR = tfRotVectorToMatrix( xpred )
  idMat = tf$linalg$matmul(  xR, xR, transpose_b=TRUE )
  deltaMat2 = tf$linalg$matmul(  xR, xpredR, transpose_b=TRUE )
  so3diff2 = idMat - deltaMat2
  frobnormexp2 = tf$reduce_sum( tf$multiply(so3diff2, so3diff2), axis=list(1L,2L) )
  tf$reduce_mean( frobnormexp2 )
  #  deltaMat1 = tf$linalg$matmul(  xpredR, xpredR, transpose_b=TRUE )
  #  so3diff1 = idMat - deltaMat1 # not necessary because we use polar decom
  #  frobnormexp1 = tf$reduce_sum( tf$multiply(so3diff1, so3diff1), axis=list(1L,2L) )
}


testingError <- function( mySdAff ) {
  with(tf$device("/cpu:0"), {
    predRot <- predict( orinet, tf$cast( ggte[[1]], mytype), batch_size = 4 )
  })
  # assess the matrix distances
  matdist = rep( NA, length( ggte[[3]] ) )
  for ( k in 1:length( ggte[[3]] ) ) {
    mm = matrix( predRot[k,], nrow=3, byrow=F)
    mmm = cbind( mm, pracma::cross( mm[,1], mm[,2] ) )
    mm = polarDecomposition( mmm )$Z
    matdist[k] = norm(  diag(3) - mm %*% t( ggte[[3]][[k]] ) , "F" )
    }
  return( mean( matdist ) )
  }



orinet = createResNetModel3D(
      list(NULL,NULL,NULL,1),
      inputScalarsSize = 0,
      numberOfClassificationLabels = 6,
      layers = 1:4,
      residualBlockSchedule = c(3, 4, 6, 3),
      lowestResolution = 64,
      cardinality = 1,
      squeezeAndExcite = TRUE,
      mode = "regression"
    )

locsds = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5,1, 2, 5 )
num_epochs <- rep( 50, length( locsds ) )
for ( locsdct in length(locsds):length(locsds) ) {
  locsd = locsds[locsdct]
  if ( locsd >= 0.5 ) num_epochs = 100
  if ( locsd >= 1.0 ) num_epochs = 40000
  mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
  opre = paste0("models/mouse_rotation_3D_GPU_update",mygpu,"locsd",locsd)
  mdlfn = paste0(opre,'.h5')
  mdlfnr = paste0(opre,'_recent.h5')
  lossperfn = paste0(opre,'_training_history.csv')
  if ( file.exists( mdlfn ) )
    load_model_weights_hdf5( orinet,  mdlfn )
  # Training loop -----------------------------------------------------------
  optimizerE <- tf$keras$optimizers$Adam(1e-7)
  lossper = data.frame( loss=NA, rotval=NA, mseval=NA, SD=NA, testErr=NA,
    runningmean=NA )
  epoch=1
  if ( file.exists( lossperfn ) ) {
    lossper = read.csv( lossperfn )
    epoch = nrow( lossper ) + 1
    }
  for (epoch in epoch:num_epochs ) {
  #  if ( epoch == 525 ) optimizerE <- tf$keras$optimizers$Adam(1e-3)
    locsd = locsds[locsdct]
  #  if ( locsd > 0.5 ) locsd = 0.5
    batchSize = 16
    gg = generateData( batch_size = batchSize,  mySdAff=locsd, verbose=FALSE  )
    datalist = list(
      tf$cast(gg[[1]],mytype), # images
      tf$cast(gg[[2]],mytype)
    )
    with(tf$GradientTape(persistent = FALSE) %as% tape, {
        rotp <- orinet( datalist[[1]] )
        mseval = tf$reduce_mean( tf$losses$MSE(  datalist[[2]], rotp  ) )
        xR = tfRotVectorToMatrix( datalist[[2]] )
        xpredR = tfRotVectorToMatrix( rotp )
        idMat = tf$linalg$matmul(  xR, xR, transpose_b=TRUE )
        deltaMat2 = tf$linalg$matmul(  xR, xpredR, transpose_b=TRUE )
        so3diff2 = idMat - deltaMat2
        frobnormexp2 = tf$reduce_sum( tf$multiply(so3diff2, so3diff2), axis=list(1L,2L) )
        rotval = tf$reduce_mean( frobnormexp2 )
        loss = rotval +  mseval * tf$constant(20.0)
      })
    rot_gradients <- tape$gradient(loss, orinet$trainable_variables)
    optimizerE$apply_gradients(purrr::transpose(list(
        rot_gradients, orinet$trainable_variables )))
    lossper[epoch,'loss']= as.numeric(loss) # tracker$metrics$loss[1]
    lossper[epoch,'rotval']= as.numeric(rotval) # tracker$metrics$loss[1]
    lossper[epoch,'mseval']= as.numeric(mseval) # tracker$metrics$loss[1]
    lossper[epoch,'SD']=locsd
    loepoch = (epoch-100)
    if ( loepoch < 1 ) loepoch = 1
    lossper[epoch,'runningmean']=mean(lossper[loepoch:(epoch-1),'loss'])
    if ( lossper[epoch,'runningmean'] < min( lossper[loepoch:(epoch-1),'runningmean'], na.rm=T  ) ) {
      print("best-recent")
  #    save_model_weights_hdf5( orinet, mdlfnr )
      }
    write.csv( lossper, lossperfn, row.names=FALSE )
    lossper[epoch,'best']=FALSE
    if ( epoch %% 5 == 1  ) {
      lossper[epoch,'testErr']=as.numeric( testingError( locsd ) )
      if ( lossper[epoch,'testErr'] < min( lossper[1:(epoch-1),'testErr'], na.rm=T ) ) {
        print("best")
        lossper[epoch,'best']=TRUE
        save_model_weights_hdf5( orinet, mdlfn )
        }
      }
    print( tail( lossper, 1 ) )
  }
} # locsd loop
