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
fns = Sys.glob( "bigdata/152-30*rec.nii.gz" )
fn = fns[1]
oimg = antsImageRead( fn )
img = denoiseImage( oimg, noiseModel='Gaussian', p = 2 ) # slower, maybe better
# img = smoothImage( oimg, 0.5, sigmaInPhysicalCoordinates=FALSE )
img = iMath( img, "TruncateIntensity", lowerTrunc, upperTrunc ) %>% iMath("Normalize")
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

iarr = array( as.array( img ), dim = c( 1, dim( img ), 1 ) )
skullout <- predict( unetSkull, tf$cast( iarr, mytype), batch_size = 1 )
segimg = as.antsImage( skullout[1,,,,1] ) %>% antsCopyImageInfo2( img )
antsImageWrite( thresholdImage(segimg,0.5,1), '/tmp/temp.nii.gz' )
print( paste( fn , "done" ) )
plot( img, segimg, nslices=21, ncol=7, axis=1 )

# now do the same in the reo space
reoTemplate = antsImageRead( "templateImage.nii.gz" ) %>% iMath("PadImage",32)
reg = antsRegistration( reoTemplate, img, 'Affine' )
imgw = reg$warpedmovout
iarr = array( as.array( imgw ), dim = c( 1, dim( imgw ), 1 ) )
skullout <- predict( unetSkull, tf$cast( iarr, mytype), batch_size = 1 )
segimg = as.antsImage( skullout[1,,,,1] ) %>% antsCopyImageInfo2( reoTemplate )
segimg = antsApplyTransforms( img, segimg, reg$fwdtransforms, whichtoinvert = TRUE )
segbig = thresholdImage(segimg,0.5,1) %>% iMath("GetLargestComponent")
antsImageWrite( segbig, '/tmp/temp2.nii.gz' )
print( paste( fn , "done" ) )
plot( img, segimg, nslices=21, ncol=7, axis=1 )
