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
postfix = c( "reorient.nii.gz", "skull.nii.gz", "pts.nii.gz", "pts.csv" )
demog = data.frame(
  ids = myids,
  reo = paste0( myids, postfix[1]),
  skull = paste0( myids, postfix[2]),
  ptsi = paste0( myids, postfix[3]),
  pts = paste0( myids, postfix[4]) )
demog$isTrain = TRUE
demog$isTrain[ sample(1:nrow( demog ), 16 ) ] = FALSE

# define an image template that lets us penalize rotations away from this reference
reoTemplate = antsImageRead( "templateImage.nii.gz" )
newspc = antsGetSpacing( reoTemplate )
reoTemplate = resampleImage( reoTemplate, newspc ) %>%
  iMath( "PadImage", 32 )%>%
  iMath( "Normalize" )
locdim = dim(reoTemplate)
lowerTrunc=1e-6
upperTrunc=0.98
numToTest = sum( demog$isTrain )
if ( ! exists( "imgList" ) ) {
  imgList = list()
  maskList = list()
  mynums = which( demog$isTrain )
  ct = 1
  for ( k in mynums ) {
    imgList[[ct]] = antsImageRead( paste0( "preprocessed/", demog$reo[k] ) )
    maskList[[ct]] = antsImageRead( paste0( "preprocessed/", demog$skull[k] ) )
    ct = ct + 1
    }
  }
##################################
if ( ! exists( "imgListTest" ) ) {
  mynums = which( !demog$isTrain )
  imgListTest = list()
  maskListTest = list()
  ct = 1
  for ( k in mynums ) {
    imgListTest[[ct]] = antsImageRead( paste0( "preprocessed/", demog$reo[k] ) )
    maskListTest[[ct]] = antsImageRead( paste0( "preprocessed/", demog$skull[k] ) )
    ct = ct + 1
    }
  }

print("done collecting")
gc()

mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
nChannelsIn = 1
nChannelsOut = 1
opre = paste0("models/mouse_skull_seg_from_ct_",mygpu)
mdlfn = paste0(opre,'.h5')
mdlfnr = paste0(opre,'_recent.h5')
lossperfn = paste0(opre,'_training_history.csv')


###############################
generateData <- function( batch_size = 32, mySdAff=5, isTest=FALSE, verbose=FALSE ) {
  if ( isTest ) {
    imgListLocal=imgListTest
    maskListLocal = maskListTest
  } else {
    imgListLocal=imgList
    maskListLocal = maskList
  }
  nBlob = nChannelsOut
  X = array( dim = c( batch_size, dim(reoTemplate), nChannelsIn ) )
  Xmask = array( dim = c( batch_size, dim(reoTemplate), 1 ) )
  for ( looper in 1:batch_size ) {
    whichSample = sample(1:length(imgListLocal),1)
    myseed = sample(1:100000000,1)
    temp = iMath(imgListLocal[[whichSample]],"Normalize")
    rr = randomAffineImage( temp, transformType='ScaleShear', sdAffine=mySdAff, seeder = myseed )
    loctx = rr[[2]]
    tempimg = applyAntsrTransformToImage( loctx, temp, reoTemplate  )
    X[looper,,,,1] = as.array( tempimg )
    tempmask = applyAntsrTransformToImage( loctx, maskListLocal[[whichSample]], reoTemplate, interpolation = "nearestNeighbor"  )
    Xmask[looper,,,,1] = as.array( tempmask )
    }
  gc()
  return( list(
    X,   # the images
    Xmask
  ) ) # input
}
########################
# make test data
ggte = generateData( batch_size=4,  mySdAff=0.1, isTest = TRUE )
# Parameters --------------------------------------------------------------
K <- keras::backend()

# Model -------------------------------------------------------------------
unetSkull = createUnetModel3D(
       list( NULL, NULL, NULL, nChannelsIn),
       numberOfOutputs = nChannelsOut,
       numberOfLayers = 4,
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

if ( file.exists( mdlfn ) ) {
  print("Load from prior training history of this model")
  load_model_weights_hdf5( unetSkull,  mdlfn )
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


testingError <- function(  ) {
  skullout <- predict( unetSkull, tf$cast( ggte[[1]], mytype), batch_size = 1 )
  timg = as.antsImage( as.array( ggte[[1]][1,,,,1] ) ) %>% antsCopyImageInfo2( reoTemplate )
  mimg = as.antsImage( as.array( ggte[[2]][1,,,,1] ) ) %>% antsCopyImageInfo2( reoTemplate )
  segimg = as.antsImage( skullout[1,,,,1] ) %>% antsCopyImageInfo2( reoTemplate )
  dicer =  binary_dice_loss( tf$cast( ggte[[2]], mytype), tf$cast( skullout, mytype) )
  return( dicer )
  }

# Training loop -----------------------------------------------------------
num_epochs <- 20000
generator_batch_size = 1
local_batch_size = 1
lossper = data.frame( loss=NA, testErr=NA, bdrunningmean=NA )
epoch=1
if ( file.exists( lossperfn ) ) {
  lossper = read.csv( lossperfn )
  epoch = nrow( lossper ) + 1
  }

optimizerE =  tf$keras$optimizers$Adam(5e-5)

for (epoch in epoch:num_epochs ) {
  locsd = ( epoch^1.15 / num_epochs ) * 5
  if ( locsd > 5 ) locsd = 5
  with(tf$device("/cpu:0"), {
    gg = generateData( batch_size = generator_batch_size,  mySdAff=locsd, verbose=FALSE  )
    datalist = list(
      tf$cast(gg[[1]],mytype), # images
      tf$cast(gg[[2]],mytype) # mask
    )
  })
  with(tf$GradientTape(persistent = FALSE) %as% tape, {
      segout <- unetSkull( datalist[[1]] )
      loss = binary_dice_loss( datalist[[2]], segout )
    })
  unet_gradients <- tape$gradient(loss, unetSkull$variables)
  optimizerE$apply_gradients(purrr::transpose(list(
      unet_gradients, unetSkull$variables )))
  lossper[epoch,'loss']=as.numeric(loss)
  lossper[epoch,'SD']=locsd
  loepoch = (epoch-100)
  if ( loepoch < 1 ) loepoch = 1
  lossper[epoch,'bdrunningmean']=mean(lossper[loepoch:(epoch-1),'loss'],na.rm=TRUE)
  if ( epoch > 50 )
    if ( lossper[epoch,'bdrunningmean'] < min( lossper[loepoch:(epoch-1),'bdrunningmean'], na.rm=T  ) ) {
      print("best-recent")
      }
  write.csv( lossper, lossperfn, row.names=FALSE )
  if ( epoch %% 50 == 1 & epoch > 50 ) {
    with(tf$device("/cpu:0"), {
      lossper[epoch,'testErr']=as.numeric( testingError( ) )
      })
    if ( lossper[epoch,'testErr'] <= min( lossper[1:epoch,'testErr'], na.rm=TRUE ) ) {
      print("best")
      save_model_weights_hdf5( unetSkull, mdlfn )
      }
    }
  print( tail( lossper, 1 ) )
  }
