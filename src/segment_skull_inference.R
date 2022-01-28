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
lowerTrunc=1e-6
upperTrunc=0.98
set.seed( 2 )
fns = Sys.glob("bigdata/skull_155_10*rec.nii.gz")
fn = fns[1]
oimg = antsImageRead( fn )
img = denoiseImage( oimg, noiseModel='Gaussian')
img = iMath( img, "TruncateIntensity", lowerTrunc, upperTrunc ) %>% iMath("Normalize")


# define an image template that lets us penalize rotations away from this reference
reoTemplate = antsImageRead( "templateImage.nii.gz" )
newspc = antsGetSpacing( reoTemplate ) # * 1.22
reoTemplate = resampleImage( reoTemplate, newspc ) %>%
  iMath( "PadImage", 32 )%>%
  iMath( "Normalize" )
locdim = dim(reoTemplate)
nChannelsIn = 1
nChannelsOut = 1
mdlfn = "models/mouse_skull_seg_from_ct_3.h5"
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
load_model_weights_hdf5( unetSkull,  mdlfn )

reg = antsRegistration( reoTemplate, img, 'Affine' )
iarr = array( as.array( reg$warpedmovout ), dim = c( 1, dim( reoTemplate ), 1 ) )
skullout <- predict( unetSkull, tf$cast( iarr, mytype), batch_size = 1 )
segimg = as.antsImage( skullout[1,,,,1] ) %>% antsCopyImageInfo2( reoTemplate ) %>%
  thresholdImage(0.5,1)
plot( reg$warpedmovout, segimg, nslices=21, ncol=7, axis=1 )
